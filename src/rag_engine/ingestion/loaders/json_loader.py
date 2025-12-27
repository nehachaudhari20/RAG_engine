# src/rag_engine/ingestion/loaders/json_loader.py

from typing import Dict, Any
import json
from rag_engine.ingestion.normalizer import normalize_document
from rag_engine.schema.document import Document


def load_json(data: Dict[str, Any], source: str = "json") -> Document:
    content = json.dumps(data, indent=2)
    return normalize_document(
        content=content,
        source=source,
        metadata={"keys": list(data.keys())}
    )
