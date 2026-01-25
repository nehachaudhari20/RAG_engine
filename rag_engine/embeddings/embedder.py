# rag_engine/embeddings/embedder.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.
        """
        return self.model.encode(texts, convert_to_numpy=True)
