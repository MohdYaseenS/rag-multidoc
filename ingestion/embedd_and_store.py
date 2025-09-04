import json
from retrieval.vectorstore import VectorStore

def load_chunks_and_store(path="chunks.json"):
    with open(path, "r") as f:
        chunks = json.load(f)
    vs = VectorStore()
    vs.store(chunks)
    print(f"✅ Stored {len(chunks)} chunks in {vs.__class__.__name__}")

if __name__ == "__main__":
    load_chunks_and_store()
