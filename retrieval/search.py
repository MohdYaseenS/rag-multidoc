import argparse
from retrieval.vectorstore import VectorStore

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
        print(f"{i}. (score={score:.4f}) {chunk[:200]}...\n")  # show preview

if __name__ == "__main__":
    main()
