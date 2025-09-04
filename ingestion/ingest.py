import fitz  # PyMuPDF
import json
from pathlib import Path
from chunker import chunk_text
import uuid

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def create_chunk_list(chunks, filename):
    chunk_list = []
    for i, chunk in enumerate(chunks):
        # Create a dictionary for each chunk with its data and metadata
        chunk_dict = {
            "id": str(uuid.uuid4()),  # Generate a unique ID
            "text": chunk,             # The actual text content
            "chunk_number": i + 1,     # 1, 2, 3...
            "source": filename         # Which document did this come from?
        }
        chunk_list.append(chunk_dict)
    return chunk_list


def save_to_json(chunk_list, output_file="chunks.json"):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_list, f, indent=4, ensure_ascii=False)
    print(f"Chunks saved to {output_file}")

if __name__ == "__main__":
    pdf_path = "C:\\Users\\YASEEN\\OneDrive\\Desktop\\Projects\\rag-multidoc\\ingestion\\pdf_documents\\test_pdf_file.pdf"
    output_file = "chunks.json"

    print("Starting ingestion process...")
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    chunk_list = create_chunk_list(chunks, pdf_path) # Create the list of dicts

    save_to_json(chunk_list, output_file) # Save to file

    print("Saved to Json")