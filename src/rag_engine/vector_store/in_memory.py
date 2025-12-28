# src/rag_engine/vector_store/in_memory.py

import numpy as np
from typing import List, Tuple
from rag_engine.vector_store.base import BaseVectorStore
from rag_engine.schema.document import Chunk


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self):
        self.vectors = []
        self.chunks = []

    def add(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("Chunk has no embedding")
            self.vectors.append(chunk.embedding)
            self.chunks.append(chunk)

    def search(self, query_vector: List[float], top_k: int):
        scores = []
        q = np.array(query_vector)

        for vec, chunk in zip(self.vectors, self.chunks):
            score = np.dot(q, vec)
            scores.append((chunk, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
