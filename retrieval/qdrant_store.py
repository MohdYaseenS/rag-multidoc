from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from ingestion.embeddings import get_embedding
from retrieval.base_store import BaseVectorStore
from api.core.config import settings

COLLECTION_NAME = "docs"

class QdrantVectorStore(BaseVectorStore):
    def __init__(self, dim=384):
        self.dim = dim
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

        # Create collection if it doesn't exist
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )

    def store(self, chunks):
        points = [
            PointStruct(
                id=i,
                vector=get_embedding(c["text"]),
                payload={"chunk": c}
            )
            for i, c in enumerate(chunks)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query, top_k=3):
        qvec = get_embedding(query)
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=qvec,
            limit=top_k
        )
        return [(r.payload["chunk"], r.score) for r in results]