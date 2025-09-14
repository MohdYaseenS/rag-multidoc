from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class BaseVectorStore(ABC):
    @abstractmethod
    def store(self, chunks: List[Dict]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        pass