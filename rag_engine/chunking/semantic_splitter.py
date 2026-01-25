# rag_engine/chunking/semantic_splitter.py
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def should_split(
    prev_embedding: np.ndarray,
    curr_embedding: np.ndarray,
    threshold: float = 0.75
) -> bool:
    """
    Returns True if topic shift detected.
    """
    sim = cosine_similarity(
        prev_embedding.reshape(1, -1),
        curr_embedding.reshape(1, -1)
    )[0][0]

    return sim < threshold
