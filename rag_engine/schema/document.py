# src/rag_engine/schema/document.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid


@dataclass
class Document:
    """
    Canonical representation of any ingested data.
    PDFs, logs, JSON, text — everything becomes a Document.
    """
    id: str
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class Chunk:
    """
    Chunked unit used for embedding and retrieval.
    """
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list] = None

    @staticmethod
    def create(
        document_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> "Chunk":
        return Chunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            content=content,
            metadata=metadata
        )
