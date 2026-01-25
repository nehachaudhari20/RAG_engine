# rag_engine/index/index_manager.py
from typing import List
import numpy as np

from rag_engine.schema.block import Block
from rag_engine.embeddings.embedder import Embedder
from rag_engine.index.vector_index import VectorIndex


class IndexManager:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.blocks: List[Block] = []
        self.index = None

    def build(self, blocks: List[Block]):
        self.blocks = blocks

        embeddings = self.embedder.embed_batch(
            [b.content for b in blocks]
        )

        dim = embeddings.shape[1]
        self.index = VectorIndex(dim)
        self.index.add(embeddings)

    def query(self, query: str, top_k: int = 5) -> List[Block]:
        query_embedding = self.embedder.embed(query)
        indices, _ = self.index.search(query_embedding, top_k)
        return [self.blocks[i] for i in indices]
