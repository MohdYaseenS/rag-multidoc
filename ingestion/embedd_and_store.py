import json
from retrieval.vector_store import VectorStore

def load_chunks_and_store(path="chunks.json"):
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    vs = VectorStore()
    vs.store(chunks)
    print(f"✅ Stored {len(chunks)} chunks in {vs.__class__.__name__}")

if __name__ == "__main__":
    load_chunks_and_store()
