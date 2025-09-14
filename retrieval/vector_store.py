from retrieval.faiss_store import FaissVectorStore
from retrieval.qdrant_store import QdrantVectorStore
from api.core.config import settings

class VectorStore:
    def __init__(self):

        if settings.VECTOR_DB == "faiss":
            self.backend = FaissVectorStore()
            
        elif settings.VECTOR_DB == "qdrant":
            self.backend = QdrantVectorStore()
        else:
            raise ValueError(f"Unsupported VECTOR_DB: {settings.VECTOR_DB}")
        
        print(f"✅ Using vector database: {settings.VECTOR_DB}")

    def store(self, chunks):
        """Store chunks in the configured backend"""
        return self.backend.store(chunks)

    def search(self, query, top_k=3):
        """Search chunks in the configured backend"""
        return self.backend.search(query, top_k)

    def clear(self):
        """Clear the vector database"""
        return self.backend.clear()