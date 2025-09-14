import argparse
import atexit

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_text
from retrieval.faiss_store import FaissStore
from llm.llm_service import LLMService


class RAGPipeline:
    def __init__(self, pdf_paths):
        # 1. Load + chunk PDFs once
        all_chunks = []
        for pdf_path in pdf_paths:
            print(f"📄 Loading: {pdf_path}")
            docs = load_pdf(pdf_path)
            chunks = chunk_text(docs, chunk_size=500, overlap=50)
            print(f"✅ Extracted {len(chunks)} chunks from {pdf_path}")
            all_chunks.extend(chunks)

        # 2. Embed + store once
        self.store = FaissStore()
        self.store.add_documents(all_chunks)
        print(f"✅ Stored {len(all_chunks)} chunks in FAISS")

        # 3. Setup LLM
        self.llm = LLMService()

    def ask(self, query, top_k=3):
        # Retrieve
        retrieved = self.store.search(query, top_k=top_k)
        print(f"🔍 Retrieved {len(retrieved)} relevant chunks")

        context = "\n".join([r.page_content for r in retrieved])

        # Generate
        prompt = f"""You are a helpful assistant.
Use the following context to answer the user query.

Context:
{context}

Question: {query}
Answer:"""

        answer = self.llm.generate(prompt)
        return answer, retrieved

    def cleanup(self):
        if hasattr(self.store, "reset"):
            self.store.reset()
            print("🧹 FAISS index cleared from memory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline on PDFs")
    parser.add_argument("--pdfs", nargs="+", required=True, help="Path(s) to PDF files")
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    pipeline = RAGPipeline(args.pdfs)
    atexit.register(pipeline.cleanup)

    # Interactive Q&A loop
    while True:
        query = input("\n❓ Enter your question (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        answer, sources = pipeline.ask(query, top_k=args.top_k)

        print("\n==============================")
        print(f"💡 Question: {query}")
        print(f"🤖 Answer: {answer}")
        print("\n📚 Sources:")
        for idx, src in enumerate(sources, 1):
            print(f"[{idx}] {src.metadata.get('source', 'unknown')}, page {src.metadata.get('page', 'N/A')}")