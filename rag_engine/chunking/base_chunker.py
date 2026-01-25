# rag_engine/chunking/base_chunker.py
from abc import ABC, abstractmethod
from typing import List
from rag_engine.schema.block import Block


class BaseChunker(ABC):

    @abstractmethod
    def chunk(self, blocks: List[Block]) -> List[Block]:
        """
        Takes paragraph-level blocks and returns chunk-level blocks.
        """
        pass
