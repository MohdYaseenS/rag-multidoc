from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations"""
    
    @abstractmethod
    def store(self, chunks: List[Dict]) -> None:
        """Store chunks in the vector database"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for similar chunks"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear the vector database"""
        pass