# rag_engine/retrieval/retriever.py
from typing import List

from rag_engine.schema.block import Block
from rag_engine.index.index_manager import IndexManager


class Retriever:
    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager

    def retrieve(self, query: str, top_k: int = 5) -> List[Block]:
        retrieved = self.index_manager.query(query, top_k)

        # Preserve document order for readability
        retrieved = sorted(retrieved, key=lambda b: b.order_index)

        return retrieved
