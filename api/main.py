from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.qa import router as qa_router
from api.core.config import settings

app = FastAPI(
    title="Multi-Document RAG API",
    description="A Retrieval Augmented Generation system for multiple documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Multi-Document RAG API",
        "status": "running",
        "vector_db": settings.VECTOR_DB,
        "embedding_provider": settings.EMBEDDING_PROVIDER
    }

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "API is working!"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "vector_db": settings.VECTOR_DB,
        "llm_provider": settings.LLM_PROVIDER,
    }

# Include routers
app.include_router(qa_router, prefix="/api/v1", tags=["QA"])

