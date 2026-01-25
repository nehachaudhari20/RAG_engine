# rag_engine/index/vector_index.py
from typing import List
import faiss
import numpy as np


class VectorIndex:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add(self, vectors: np.ndarray):
        """
        vectors: shape (N, D)
        """
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int):
        """
        query_vector: shape (D,)
        """
        distances, indices = self.index.search(
            query_vector.reshape(1, -1), top_k
        )
        return indices[0], distances[0]
