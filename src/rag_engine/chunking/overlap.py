# src/rag_engine/chunking/overlap.py

from typing import List
from rag_engine.chunking.base import BaseChunker
from rag_engine.schema.document import Document, Chunk


class OverlapChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        assert overlap < chunk_size
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]
            if not chunk_text:
                continue
            chunks.append(
                Chunk.create(
                    document_id=document.id,
                    content=chunk_text,
                    metadata=document.metadata
                )
            )
        return chunks
