# src/rag_engine/ingestion/loaders/text_loader.py

from rag_engine.ingestion.normalizer import normalize_document
from rag_engine.schema.document import Document


def load_text(text: str, source: str = "text") -> Document:
    return normalize_document(
        content=text,
        source=source
    )
