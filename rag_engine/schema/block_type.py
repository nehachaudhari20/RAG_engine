# src/rag_engine/schema/retrieval.py

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class RetrievalResult:
    """
    Single retrieved chunk with score.
    """
    chunk_id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]


@dataclass
class RetrievalResponse:
    """
    Structured output of a retrieval operation.
    """
    query: str
    results: List[RetrievalResult]
