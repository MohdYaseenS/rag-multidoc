from api.core.config import settings

from retrieval.faiss_store import FaissVectorStore
from retrieval.qdrant_store import QdrantVectorStore

def VectorStore():
    if settings.VECTOR_DB == "faiss":
        return FaissVectorStore()
    elif settings.VECTOR_DB == "qdrant":
        return QdrantVectorStore()
    else:
        raise ValueError(f"Unknown VECTOR_DB: {settings.VECTOR_DB}")