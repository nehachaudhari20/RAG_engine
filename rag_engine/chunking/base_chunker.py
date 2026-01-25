# src/rag_engine/chunking/base.py

from abc import ABC, abstractmethod
from typing import List
from rag_engine.schema.document import Document, Chunk


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        pass
