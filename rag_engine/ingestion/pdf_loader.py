# src/rag_engine/ingestion/normalizer.py

from typing import Dict, Any, Optional
from datetime import datetime
from rag_engine.schema.document import Document
import uuid


def normalize_document(
    content: str,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[datetime] = None
) -> Document:
    """
    Convert raw input into a canonical Document.
    """
    return Document(
        id=str(uuid.uuid4()),
        content=content,
        source=source,
        metadata=metadata or {},
        timestamp=timestamp
    )
