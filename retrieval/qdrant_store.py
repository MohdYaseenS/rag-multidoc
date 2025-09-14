from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from ingestion.embedding import get_embedding
from retrieval.base_store import BaseVectorStore
from api.core.config import settings

COLLECTION_NAME = "document_chunks"

class QdrantVectorStore(BaseVectorStore):
    def __init__(self):
        self.dim = 384 if settings.EMBEDDING_PROVIDER == "local" else 1536
        self.client = QdrantClient(
            host=settings.QDRANT_HOST, 
            port=settings.QDRANT_PORT
        )
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            existing_names = [collection.name for collection in collections.collections]
            
            if COLLECTION_NAME not in existing_names:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.dim, 
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created Qdrant collection: {COLLECTION_NAME}")
        except Exception as e:
            raise ValueError(f"Error ensuring Qdrant collection: {str(e)}")

    def store(self, chunks):
        """Store chunks in Qdrant"""
        if not chunks:
            raise ValueError("No chunks to store")
        
        points = []
        for i, chunk in enumerate(chunks):  # ← CHANGED: Use numeric index
            if not chunk.get("id") or not chunk.get("text"):
                continue
                
            point = PointStruct(
                id=i,  # ← CHANGED: Use index instead of chunk["id"]
                vector=get_embedding(chunk["text"]),
                payload=chunk
            )
            points.append(point)
        
        if not points:
            raise ValueError("No valid chunks to store")
        
        # Upsert points
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )
        
        print(f"✅ Stored {len(points)} chunks in Qdrant")

    def search(self, query, top_k=3):
        """Search for similar chunks in Qdrant"""
        if not query or not query.strip():
            return []
        
        try:
            qvec = get_embedding(query)
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=qvec,
                limit=top_k,
                with_payload=True
            )
            
            return [(hit.payload, hit.score) for hit in results]
            
        except Exception as e:
            raise ValueError(f"Qdrant search error: {str(e)}")

    def clear(self):
        """Clear the collection"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            print(f"✅ Deleted Qdrant collection: {COLLECTION_NAME}")
            
            # Recreate collection
            self._ensure_collection()
            
        except Exception as e:
            raise ValueError(f"Error clearing Qdrant collection: {str(e)}")