# rag_engine/ingestion/text_normalizer.py
import re

def normalize_text(text: str) -> str:
    # Fix missing spaces between lowercase-uppercase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Fix missing spaces after punctuation
    text = re.sub(r"([.,;:])([A-Za-z])", r"\1 \2", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()
