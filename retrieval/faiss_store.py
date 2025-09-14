import json
import hashlib
import numpy as np
import faiss
from ingestion.embedding import get_embedding
from retrieval.base_store import BaseVectorStore

class FaissVectorStore(BaseVectorStore):
    def __init__(self, dim=384, index_path="faiss.index", map_path="chunks_map.json"):
        self.dim = dim
        self.index_path = index_path
        self.map_path = map_path

    def _get_numeric_id(self, text_id: str) -> int:
        return int(hashlib.md5(text_id.encode()).hexdigest(), 16) % (2**63 - 1)

    def store(self, chunks):
        index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))
        vectors = np.array([get_embedding(c["text"]) for c in chunks]).astype("float32")
        ids = [self._get_numeric_id(c["id"]) for c in chunks]
        index.add_with_ids(vectors, np.array(ids, dtype="int64"))

        # Save index
        faiss.write_index(index, self.index_path)

        # Save ID to chunk map
        id_map = {str(self._get_numeric_id(c["id"])): c for c in chunks}
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(id_map, f, indent=2, ensure_ascii=False)

    def search(self, query, top_k=3):
        index = faiss.read_index(self.index_path)
        with open(self.map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)

        qvec = np.array([get_embedding(query)]).astype("float32")
        distances, ids = index.search(qvec, top_k)

        results = []
        for j, idx in enumerate(ids[0]):
            chunk = id_map.get(str(idx))
            if chunk:
                results.append((chunk, float(distances[0][j])))
        return results
