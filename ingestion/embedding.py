import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from api.core.config import settings

# Cache models and clients
_local_model = None
_openai_client = None

def get_embedding_dim():
    """Return the correct embedding dimension based on provider"""
    if settings.EMBEDDING_PROVIDER == "local":
        return 384  # all-MiniLM-L6-v2 dimension
    elif settings.EMBEDDING_PROVIDER == "openai":
        return 1536  # text-embedding-3-small dimension
    else:
        raise ValueError(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}")

def get_embedding(text: str):
    """
    Generate embedding for text using configured provider.
    
    Args:
        text (str): Input text to embed
    
    Returns:
        np.array: Embedding vector
    """
    global _local_model, _openai_client
    
    if not text or not text.strip():
        raise ValueError("Text cannot be empty for embedding")
    
    if settings.EMBEDDING_PROVIDER == "local":
        if _local_model is None:
            _local_model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = _local_model.encode([text])[0]
        return np.array(emb, dtype="float32")
    
    elif settings.EMBEDDING_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        
        if _openai_client is None:
            _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        try:
            resp = _openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            emb = resp.data[0].embedding
            return np.array(emb, dtype="float32")
        except Exception as e:
            raise ValueError(f"OpenAI embedding error: {str(e)}")
    
    else:
        raise ValueError(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}")