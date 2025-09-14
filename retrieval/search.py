import argparse
from retrieval.vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Search FAISS/Qdrant index")
    parser.add_argument("query", type=str, help="Your search query")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()

    vs = VectorStore()
    results = vs.search(args.query, top_k=args.top_k)

    print("\n🔎 Query:", args.query)
    print("Top Results:\n")
    for i, (chunk, score) in enumerate(results, start=1):
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        source = chunk.get("source", "unknown") if isinstance(chunk, dict) else "N/A"
        chunk_num = chunk.get("chunk_number", "N/A") if isinstance(chunk, dict) else "N/A"

        print(f"{i}. (score={score:.4f}) [source={source}, chunk={chunk_num}] {text[:200]}...\n")

if __name__ == "__main__":
    main()
