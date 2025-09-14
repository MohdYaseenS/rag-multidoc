import fitz  # PyMuPDF
import json
import uuid
from pathlib import Path
from ingestion.chunker import chunk_text

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF using PyMuPDF.
    
    Args:
        pdf_path (str): Path to PDF file
    
    Returns:
        str: Extracted text
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF {pdf_path}: {str(e)}")

def create_chunk_list(chunks, filename, metadata=None):
    """
    Create a list of chunk dictionaries with metadata.
    
    Args:
        chunks (list): List of text chunks
        filename (str): Source filename
        metadata (dict): Additional metadata
    
    Returns:
        list: List of chunk dictionaries
    """
    chunk_list = []
    metadata = metadata or {}
    
    for i, chunk in enumerate(chunks):
        chunk_dict = {
            "id": str(uuid.uuid4()),
            "text": chunk,
            "chunk_number": i + 1,
            "source": filename,
            **metadata
        }
        chunk_list.append(chunk_dict)
    
    return chunk_list

def save_to_json(chunk_list, output_file="chunks.json"):
    """
    Save chunks to JSON file.
    
    Args:
        chunk_list (list): List of chunk dictionaries
        output_file (str): Output JSON file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_list, f, indent=4, ensure_ascii=False)
        print(f"✅ Chunks saved to {output_file}")
    except Exception as e:
        raise ValueError(f"Error saving to JSON: {str(e)}")

def process_pdf(pdf_path, output_file=None, chunk_size=500, chunk_overlap=50):
    """
    Complete PDF processing pipeline.
    
    Args:
        pdf_path (str): Path to PDF file
        output_file (str): Output JSON file path
        chunk_size (int): Chunk size
        chunk_overlap (int): Chunk overlap
    
    Returns:
        list: Processed chunks
    """
    print(f"📄 Processing {pdf_path}...")
    
    # Extract text
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"   Extracted {len(raw_text)} characters")
    
    # Chunk text
    chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
    print(f"   Created {len(chunks)} chunks")
    
    # Create chunk list
    chunk_list = create_chunk_list(chunks, str(pdf_path))
    
    # Save to JSON if output file specified
    if output_file:
        save_to_json(chunk_list, output_file)
    
    return chunk_list

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Ingestion Pipeline")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--output", "-o", default="chunks.json", help="Output JSON file")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Chunk overlap")
    
    args = parser.parse_args()
    
    try:
        process_pdf(args.pdf_path, args.output, args.chunk_size, args.chunk_overlap)
        print("✅ Ingestion completed successfully!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit(1)