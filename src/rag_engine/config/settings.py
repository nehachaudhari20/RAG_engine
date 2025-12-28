# src/rag_engine/config/settings.py

from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    overlap: int = 100


@dataclass
class EmbeddingConfig:
    model_name: str = "local-default"
    dim: int = 384


@dataclass
class RetrievalConfig:
    top_k: int = 5


@dataclass
class RAGConfig:
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
