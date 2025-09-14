from api.core.config import settings
from .llm_providers import OpenAIService, HuggingFaceService, MockLLMService

class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        
        if self.provider == "openai":
            self.service = OpenAIService()
        elif self.provider == "huggingface":
            self.service = HuggingFaceService()
        else:
            self.service = MockLLMService()
        
        print(f"✅ Using LLM provider: {self.provider}")

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        return self.service.generate_answer(query, context_chunks)

    def is_available(self):
        return self.provider in ["openai", "huggingface"]
