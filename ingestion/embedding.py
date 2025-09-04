import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from api.core.config import settings

# Cache model(s)
_local_model = None
_openai_client = None

def get_embedding(text: str):
    global _local_model, _openai_client

    if settings.EMBEDDING_PROVIDER == "local":
        if _local_model is None:
            _local_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
        emb = _local_model.encode([text])[0]
        return np.array(emb, dtype="float32")

    elif settings.EMBEDDING_PROVIDER == "openai":
        if _openai_client is None:
            _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        resp = _openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        emb = resp.data[0].embedding
        return np.array(emb, dtype="float32")

    else:
        raise ValueError(f"Unknown embedding provider: {settings.EMBEDDING_PROVIDER}")