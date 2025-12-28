# src/rag_engine/vector_store/base.py

from abc import ABC, abstractmethod
from typing import List, Tuple
from rag_engine.schema.document import Chunk


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[Tuple[Chunk, float]]:
        pass
