import json
from retrieval.vector_store import VectorStore

def load_chunks_and_store(json_path="chunks.json"):
    """
    Load chunks from JSON and store in vector database.
    
    Args:
        json_path (str): Path to JSON file with chunks
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        if not chunks:
            print("❌ No chunks found in JSON file")
            return
        
        vs = VectorStore()
        vs.store(chunks)
        print(f"✅ Stored {len(chunks)} chunks in vector database")
        
    except FileNotFoundError:
        print(f"❌ JSON file not found: {json_path}")
    except Exception as e:
        print(f"❌ Error storing chunks: {str(e)}")

def ingest_from_json(json_path="chunks.json"):
    """
    Complete ingestion pipeline from JSON to vector store.
    """
    print(f"📦 Loading chunks from {json_path}...")
    load_chunks_and_store(json_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Embed and Store Pipeline")
    parser.add_argument("--input", "-i", default="chunks.json", help="Input JSON file")
    
    args = parser.parse_args()
    ingest_from_json(args.input)