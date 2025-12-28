# src/rag_engine/vector_store/faiss_store.py

import faiss
import numpy as np
from typing import List, Tuple

from rag_engine.vector_store.base import BaseVectorStore
from rag_engine.schema.document import Chunk


class FaissVectorStore(BaseVectorStore):
    def __init__(self, dim: int):
        """
        dim: embedding dimension (e.g. 384)
        """
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[Chunk] = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, chunks: List[Chunk]) -> None:
        vectors = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError("Chunk has no embedding")
            vectors.append(chunk.embedding)
            self.chunks.append(chunk)

        vectors = np.array(vectors).astype("float32")
        vectors = self._normalize(vectors)

        self.index.add(vectors)

    def search(
        self,
        query_vector: List[float],
        top_k: int
    ) -> List[Tuple[Chunk, float]]:

        q = np.array([query_vector]).astype("float32")
        q = self._normalize(q)

        scores, indices = self.index.search(q, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))

        return results
