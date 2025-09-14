from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Split text into chunks using recursive character splitting.
    
    Args:
        text (str): Input text to split
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    if not text or not text.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.create_documents([text])
    return [chunk.page_content for chunk in chunks]