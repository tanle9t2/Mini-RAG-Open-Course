from typing import List, Literal, Optional
from langchain.schema import HumanMessage, AIMessage
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_qa import ask_question

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or ["*"] for all origins (not recommended in prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatHistoryItem(BaseModel):
    human: str
    ai: str

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatHistoryItem]] = []

class QueryResponse(BaseModel):
    answer: str
    context: List[str]


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        # Convert chat_history to list of dicts (if needed by your ask_question)
        chat_history = [{"human": item.human, "ai": item.ai} for item in request.chat_history]

        result = ask_question(request.question, chat_history=chat_history)

        return QueryResponse(
            answer=result["answer"],
            context=result["context"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)