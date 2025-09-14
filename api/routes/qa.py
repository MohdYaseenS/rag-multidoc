from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from retrieval.vector_store import VectorStore
from llm.llm_service import LLMService
from api.core.config import settings

router = APIRouter()
vs = VectorStore()

# Initialize LLM service on demand to avoid startup issues
def get_llm_service():
    return LLMService()

class AskRequest(BaseModel):
    query: str
    top_k: int = 3

class AskResponse(BaseModel):
    answer: str
    context: list
    sources: list

@router.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    try:
        # Step 1: Retrieve relevant chunks
        results = vs.search(req.query, top_k=req.top_k)
        if not results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        chunks = [chunk for chunk, score in results]
        scores = [score for chunk, score in results]
        
        # Step 2: Generate answer
        context_texts = [chunk["text"] for chunk in chunks]
        llm_service = get_llm_service()
        answer = llm_service.generate_answer(req.query, context_texts)
        
        # Step 3: Prepare response
        sources = []
        for chunk, score in zip(chunks, scores):
            sources.append({
                "score": float(score),
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "source": chunk.get("source", "unknown"),
                "chunk_number": chunk.get("chunk_number", "N/A")
            })
        
        return AskResponse(
            answer=answer,
            context=context_texts,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.get("/health")
def health_check():
    llm_service = get_llm_service()
    return {
        "status": "healthy", 
        "vector_db": settings.VECTOR_DB,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_available": llm_service.is_available()
    }