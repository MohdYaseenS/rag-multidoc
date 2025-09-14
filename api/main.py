from fastapi import FastAPI
from api.routes import qa
from pydantic import BaseModel

from retrieval.search import search_query
from llm.llm_service import LLMService

app = FastAPI(title="Multi-Doc RAG")

@app.get("/ping")
def ping():
    return {"status": "ok"}

app.include_router(qa.router, prefix="")

app = FastAPI()
llm = LLMService(model_name="gpt-4o-mini")

class AskRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/ask")
def ask(req: AskRequest):
    # Step 1: Retrieve top chunks
    results = search_query(req.query, top_k=req.top_k)

    chunks = [chunk for _, chunk in results]

    # Step 2: Generate answer from LLM
    answer = llm.generate_answer(req.query, chunks)

    return {
        "query": req.query,
        "answer": answer,
        "sources": [{"score": score, "preview": chunk[:200]} for score, chunk in results]
    }