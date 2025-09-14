from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import os

load_dotenv()

class Settings(BaseSettings):
    # Existing settings
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    VECTOR_DB: str = os.getenv("VECTOR_DB", "faiss")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Enhanced LLM settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "huggingface")  # huggingface, openai, mock
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    class Config:
        env_file = ".env"

settings = Settings()
