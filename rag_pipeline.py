import argparse
import atexit

from ingestion.ingest import process_pdf as load_pdf

from retrieval.faiss_store import FaissVectorStore as FaissStore
from llm.llm_service import LLMService
import os



class RAGPipeline:
    def __init__(self, pdf_paths):
        # 1. Load + chunk PDFs once
        all_chunks = []
        for pdf_path in pdf_paths:
            print(f"📄 Loading: {pdf_path}")
            chunks = load_pdf(pdf_path)
            print(f"✅ Extracted {len(chunks)} chunks from {pdf_path}")
            all_chunks.extend(chunks)

        # 2. Embed + store once
        self.store = FaissStore()
        self.store.store(all_chunks)
        print(f"✅ Stored {len(all_chunks)} chunks in FAISS")

        # 3. Setup LLM
        self.llm = LLMService()

    def ask(self, query, top_k=3):
        # Retrieve
        retrieved = self.store.search(query, top_k=top_k)

        print(f"🔍 Retrieved {len(retrieved)} relevant chunks")

        context = "\n".join(doc['text'] for doc, _ in retrieved)

        answer = self.llm.generate_answer(query, context)
        return answer

    def cleanup(self):
        if hasattr(self.store, "reset"):
            self.store.reset()
            print("🧹 FAISS index cleared from memory.")

        files_to_delete = [
            "C:\\Users\\YASEEN\\OneDrive\\Desktop\\Projects\\rag-multidoc\\chunks_map.json",
            "C:\\Users\\YASEEN\\OneDrive\\Desktop\\Projects\\rag-multidoc\\faiss.index"
        ]

        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️ Deleted: {file_path}")
                else:
                    print(f"⚠️ File not found (skipped): {file_path}")
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")


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

        answer = pipeline.ask(query, top_k=args.top_k)

        print("\n==============================")
        print(f"💡 Question: {query}")
        print(f"🤖 Answer: {answer}")