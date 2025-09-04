import json
import numpy as np
import faiss
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from api.core.config import settings
from ingestion.embeddings import get_embedding

COLLECTION_NAME = "docs"

# --- Embedding Function ---
_local_model = None
_openai_client = None


# --- VectorStore Class ---
class VectorStore:
    def __init__(self, dim=384):
        self.dim = dim
        if settings.VECTOR_DB == "faiss":
            self.index_path = "faiss.index"
            self.map_path = "chunks_map.json"
        else:
            self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    def store(self, chunks):
        if settings.VECTOR_DB == "faiss":
            self._store_faiss(chunks)
        else:
            self._store_qdrant(chunks)

    def search(self, query, top_k=3):
        if settings.VECTOR_DB == "faiss":
            return self._search_faiss(query, top_k)
        else:
            return self._search_qdrant(query, top_k)

    # --- FAISS ---
    def _store_faiss(self, chunks):
        index = faiss.IndexFlatL2(self.dim)
        vectors = np.array([get_embedding(c) for c in chunks]).astype("float32")
        index.add(vectors)
        faiss.write_index(index, self.index_path)
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    def _search_faiss(self, query, top_k):
        index = faiss.read_index(self.index_path)
        with open(self.map_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        qvec = np.array([get_embedding(query)]).astype("float32")
        distances, indices = index.search(qvec, top_k)
        return [(chunks[i], float(distances[0][j])) for j, i in enumerate(indices[0])]

    # --- Qdrant ---
    def _store_qdrant(self, chunks):
        # create collection if not exists
        if COLLECTION_NAME not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )
        points = [
            PointStruct(id=i, vector=get_embedding(c), payload={"text": c})
            for i, c in enumerate(chunks)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def _search_qdrant(self, query, top_k):
        qvec = get_embedding(query)
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=qvec,
            limit=top_k
        )
        return [(r.payload["text"], r.score) for r in results]
