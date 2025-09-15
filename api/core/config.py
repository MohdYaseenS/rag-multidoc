from pydantic_settings import BaseSettings
import os


from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    EMBEDDING_PROVIDER: str = "local"
    VECTOR_DB: str = "faiss"
    OPENAI_API_KEY: str
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    LLM_PROVIDER: str = "huggingface"
    HUGGINGFACE_API_KEY: str = ""
    HUGGINGFACE_MODEL: str = "meta-llama/Llama-2-7b-chat-hf"
    OPENAI_MODEL: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"

settings = Settings()
