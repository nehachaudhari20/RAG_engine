# src/rag_engine/ingestion/loaders/log_loader.py

from datetime import datetime
from rag_engine.ingestion.normalizer import normalize_document
from rag_engine.schema.document import Document


def load_log(
    message: str,
    level: str,
    service: str,
    timestamp: datetime
) -> Document:
    return normalize_document(
        content=message,
        source="log",
        metadata={
            "level": level,
            "service": service
        },
        timestamp=timestamp
    )
