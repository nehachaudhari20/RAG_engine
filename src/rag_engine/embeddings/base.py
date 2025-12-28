# src/rag_engine/embeddings/base.py

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingModel(ABC):
    """
    Abstract embedding model interface.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents/chunks.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        """
        pass
