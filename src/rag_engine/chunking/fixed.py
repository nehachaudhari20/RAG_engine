# src/rag_engine/chunking/fixed.py

from typing import List
from rag_engine.chunking.base import BaseChunker
from rag_engine.schema.document import Document, Chunk


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        chunks: List[Chunk] = []

        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i:i + self.chunk_size]
            chunks.append(
                Chunk.create(
                    document_id=document.id,
                    content=chunk_text,
                    metadata=document.metadata
                )
            )
        return chunks
