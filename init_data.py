import os

import numpy as np
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

from langchain_experimental.text_splitter import SemanticChunker
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


def fetch_mysql_data():
    engine = create_engine(os.getenv("MYSQL_URI"))
    with engine.connect() as conn:
        query = text("""
            SELECT 
                CONCAT(u.first_name,' ', u.last_name) AS teacher_name,
                c.id AS course_id,
                c.name AS course_name,
                c.description,
                cat.name AS category,
                c.price as course_price,
                s.id AS section_id,
                s.name AS section_name,
                ct.id AS content_id,
                ct.name AS content_name
            FROM course c
            LEFT JOIN user u ON u.id = c.teacher_id
            LEFT JOIN category cat ON c.category_id = cat.id
            LEFT JOIN section s ON s.course_id = c.id
            LEFT JOIN content ct ON ct.section_id = s.id
            WHERE c.description IS NOT NULL AND c.description != ''
        """)
        result = conn.execute(query)
        return [
            {
                "course": row.course_name,
                "section": row.section_name,
                "content_name": row.content_name,
                "teacher": row.teacher_name,
                "category": row.category,
                "description": row.description,
                "price": int(row.course_price),
            }
            for row in result.fetchall()
        ]


def semantic_chunking(text, chunk_size=500, chunk_overlap=50, similarity_threshold=0.85):
    # Step 1: Rough character splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    initial_chunks = splitter.split_text(text)

    # Step 2: Embed chunks
    embeddings = embedding_model.embed_documents(initial_chunks)

    merged_chunks = []
    i = 0
    while i < len(initial_chunks):
        current_chunk = initial_chunks[i]
        current_emb = embeddings[i]

        j = i + 1
        while j < len(initial_chunks):
            sim = cosine_similarity([current_emb], [embeddings[j]])[0][0]
            if sim >= similarity_threshold:
                # Merge similar chunks
                current_chunk += " " + initial_chunks[j]
                # Average embeddings
                current_emb = np.mean([current_emb, embeddings[j]], axis=0)
                j += 1
            else:
                break

        merged_chunks.append(current_chunk)
        i = j

    return merged_chunks


def index_to_pinecone():
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = os.getenv("PINECONE_INDEX_NAME")
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # for OpenAI embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Get index object
    index = pc.Index(name=index_name)

    # Fetch and split data
    rows = fetch_mysql_data()
    docs = []
    for row in rows:
        course = row.get("course") or ""
        section = row.get("section") or ""
        content_name = row.get("content_name") or ""
        teacher = row.get("teacher") or ""
        category = row.get("category") or ""
        price = row.get("price") or 0
        description = row.get("description") or ""

        full_content = f"""
        Course: {course}
        Section: {section}
        Content: {content_name}
        Teacher: {teacher}
        Category: {category}
        Price: {price} VND
        Description: {description}
        """

        # Split into semantically meaningful chunks
        chunks = semantic_chunking(full_content)
        for chunk in chunks:
            # Build document with metadata
            doc = {
                "page_content": chunk,
                "metadata": {
                    "course": course,
                    "section": section,
                    "content_name": content_name,
                    "teacher": teacher,
                    "category": category,
                    "price": price,
                }
            }
            docs.append(doc)

    # Upload documents to Pinecone via LangChain VectorStore
    PineconeVectorStore.from_texts(
        texts=[doc["page_content"] for doc in docs],
        embedding=embedding_model,
        metadatas=[doc["metadata"] for doc in docs],
        index_name=index_name,
    )


if __name__ == "__main__":
    index_to_pinecone()
