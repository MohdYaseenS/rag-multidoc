#!/usr/bin/env python3
"""
Multi-Document RAG CLI - Complete pipeline from ingestion to Q&A

Usage:
  python rag_cli.py ingest <pdf_files>...
  python rag_cli.py query "your question"
  python rag_cli.py api
"""

import argparse
import json
import sys
from pathlib import Path
import uvicorn

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from ingestion.ingest import process_pdf, save_to_json
from retrieval.vector_store import VectorStore
from llm.llm_service import LLMService
from api.core.config import settings


def ingest_documents(pdf_paths, output_file=None, chunk_size=500, chunk_overlap=50):
    """Ingest PDF documents and store in vector database"""
    all_chunks = []
    
    for pdf_path in pdf_paths:
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            print(f"❌ File not found: {pdf_path}")
            continue
        
        try:
            chunks = process_pdf(
                str(path_obj), 
                None,  # Don't save individual files
                chunk_size, 
                chunk_overlap
            )
            all_chunks.extend(chunks)
            print(f"✅ Processed {path_obj.name}: {len(chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error processing {path_obj.name}: {str(e)}")
            continue
    
    if not all_chunks:
        print("❌ No chunks were processed successfully")
        return False
    
    # Save combined chunks to JSON if requested
    if output_file:
        try:
            save_to_json(all_chunks, output_file)
            print(f"✅ Saved {len(all_chunks)} chunks to {output_file}")
        except Exception as e:
            print(f"❌ Error saving to JSON: {str(e)}")
    
    # Store in vector database
    try:
        vs = VectorStore()
        vs.store(all_chunks)
        print(f"✅ Stored {len(all_chunks)} chunks in vector database")
        return True
        
    except Exception as e:
        print(f"❌ Error storing in vector database: {str(e)}")
        return False

def query_rag(query, top_k=3):
    """Query the RAG system and return answer with sources"""
    try:
        # Retrieve relevant chunks
        vs = VectorStore()
        results = vs.search(query, top_k=top_k)
        
        if not results:
            return "No relevant information found.", []
        
        chunks = [chunk for chunk, score in results]
        scores = [score for chunk, score in results]
        
        # Generate answer
        llm = LLMService()
        answer = llm.generate_answer(query, [chunk["text"] for chunk in chunks])
        
        # Prepare sources with scores
        sources = []
        for chunk, score in zip(chunks, scores):
            sources.append({
                "score": float(score),
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "source": chunk.get("source", "unknown"),
                "chunk_number": chunk.get("chunk_number", "N/A")
            })
        
        return answer, sources
        
    except Exception as e:
        return f"Error: {str(e)}", []

def run_api():
    """Start the FastAPI server"""
    print("🚀 Starting RAG API server...")
    print(f"   Vector DB: {settings.VECTOR_DB}")
    print(f"   Embedding: {settings.EMBEDDING_PROVIDER}")
    print("   API available at: http://localhost:8000")
    print("   Docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Document RAG CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_cli.py ingest document1.pdf document2.pdf
  python rag_cli.py query "What is machine learning?"
  python rag_cli.py api
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF documents")
    ingest_parser.add_argument("pdf_files", nargs="+", help="PDF files to ingest")
    ingest_parser.add_argument("--output", "-o", help="Output JSON file")
    ingest_parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size")
    ingest_parser.add_argument("--chunk_overlap", type=int, default=50, help="Chunk overlap")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", help="Your question")
    query_parser.add_argument("--top_k", type=int, default=3, help="Number of results to return")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the vector database")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        success = ingest_documents(
            args.pdf_files, 
            args.output, 
            args.chunk_size, 
            args.chunk_overlap
        )
        exit(0 if success else 1)
        
    elif args.command == "query":
        answer, sources = query_rag(args.query, args.top_k)
        
        print(f"\n🤖 Question: {args.query}")
        print(f"✅ Answer: {answer}")
        
        if sources:
            print(f"\n📚 Sources (top {len(sources)}):")
            for i, source in enumerate(sources, 1):
                print(f"{i}. Score: {source['score']:.4f}")
                print(f"   From: {source['source']} (chunk {source['chunk_number']})")
                print(f"   Text: {source['text']}")
                print()
        else:
            print("\n❌ No sources found")
            
    elif args.command == "api":
        run_api()
        
    elif args.command == "clear":
        try:
            vs = VectorStore()
            vs.clear()
            print("✅ Vector database cleared")
        except Exception as e:
            print(f"❌ Error clearing database: {str(e)}")
            exit(1)
            
    else:
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()