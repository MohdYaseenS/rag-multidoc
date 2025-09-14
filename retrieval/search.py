import argparse
from .vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Search FAISS/Qdrant index")
    parser.add_argument("query", type=str, help="Your search query")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    
    args = parser.parse_args()
    
    try:
        vs = VectorStore()
        results = vs.search(args.query, top_k=args.top_k)
        
        print(f"\n🔎 Query: {args.query}")
        print(f"📊 Found {len(results)} results:\n")
        
        for i, (chunk, score) in enumerate(results, start=1):
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")
            chunk_num = chunk.get("chunk_number", "N/A")
            
            preview = text[:200] + "..." if len(text) > 200 else text
            
            print(f"{i}. Score: {score:.4f}")
            print(f"   Source: {source} (chunk {chunk_num})")
            print(f"   Text: {preview}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
