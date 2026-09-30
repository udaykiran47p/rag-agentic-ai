import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME

app = FastAPI(
    title="Agentic AI Grounded RAG Chatbot API",
    description="Grounded QA Bot using LangGraph, Pinecone, and OpenAI",
    version="1.0.0",
)

# Initialize compiled LangGraph workflow
graph = build_rag_graph(index_name=PINECONE_INDEX_NAME)

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is Agentic AI according to the eBook?")

class QueryResponse(BaseModel):
    final_answer: str
    retrieved_context: List[str]
    confidence_score: float

@app.get("/health")
def health_check():
    return {"status": "ok", "index": PINECONE_INDEX_NAME}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0,
    }

    try:
        result = graph.invoke(initial_state)
        return QueryResponse(
            final_answer=result.get("answer", ""),
            retrieved_context=result.get("context", []),
            confidence_score=result.get("score", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)