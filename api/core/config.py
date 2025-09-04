import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")   # "local" | "openai"
    VECTOR_DB: str = os.getenv("VECTOR_DB", "faiss")                     # "faiss" | "qdrant"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

settings = Settings()


