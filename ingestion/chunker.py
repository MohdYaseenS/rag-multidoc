from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    # Initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    # Split the text! This creates "Document" objects.
    chunks = text_splitter.create_documents([text])
    # Extract just the text from each chunk
    chunks = [chunk.page_content for chunk in chunks]
    return chunks