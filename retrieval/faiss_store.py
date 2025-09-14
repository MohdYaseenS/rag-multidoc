import json
import hashlib
import numpy as np
import faiss
from pathlib import Path
from .base_store import BaseVectorStore
from ingestion.embedding import get_embedding
from api.core.config import settings

class FaissVectorStore(BaseVectorStore):
    def __init__(self, index_path="faiss.index", map_path="chunks_map.json"):
        self.dim = 384 if settings.EMBEDDING_PROVIDER == "local" else 1536
        self.index_path = index_path
        self.map_path = map_path
        self.index = None
        self.id_map = {}

    def _get_numeric_id(self, text_id: str) -> int:
        """Convert text ID to numeric ID for FAISS"""
        return int(hashlib.md5(text_id.encode()).hexdigest(), 16) % (2**63 - 1)

    def _load_index(self):
        """Load index from file or create new"""
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
            if Path(self.map_path).exists():
                with open(self.map_path, "r", encoding="utf-8") as f:
                    self.id_map = json.load(f)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))
            self.id_map = {}

    def store(self, chunks):
        """Store chunks in FAISS index"""
        self._load_index()
        
        if not chunks:
            raise ValueError("No chunks to store")
        
        # Prepare vectors and IDs
        vectors = []
        ids = []
        new_id_map = {}
        
        for chunk in chunks:
            if not chunk.get("id") or not chunk.get("text"):
                continue
                
            vector = get_embedding(chunk["text"])
            numeric_id = self._get_numeric_id(chunk["id"])
            
            vectors.append(vector)
            ids.append(numeric_id)
            new_id_map[str(numeric_id)] = chunk
        
        if not vectors:
            raise ValueError("No valid chunks to store")
        
        # Convert to numpy arrays
        vectors_np = np.array(vectors).astype("float32")
        ids_np = np.array(ids, dtype="int64")
        
        # Add to index
        self.index.add_with_ids(vectors_np, ids_np)
        
        # Update ID map
        self.id_map.update(new_id_map)
        
        # Save to files
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Stored {len(vectors)} chunks in FAISS")

    def search(self, query, top_k=3):
        """Search for similar chunks"""
        if self.index is None:
            self._load_index()
        
        if not query or not query.strip():
            return []
        
        # Generate query embedding
        qvec = np.array([get_embedding(query)]).astype("float32")
        
        # Search
        distances, ids = self.index.search(qvec, top_k)
        
        results = []
        for j, idx in enumerate(ids[0]):
            if idx != -1:  # FAISS returns -1 for missing results
                chunk = self.id_map.get(str(idx))
                if chunk:
                    results.append((chunk, float(distances[0][j])))
        
        return results

    def clear(self):
        """Clear the index"""
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))
        self.id_map = {}
        
        # Remove files
        for path in [self.index_path, self.map_path]:
            if Path(path).exists():
                Path(path).unlink()
        
        print("✅ FAISS index cleared")
