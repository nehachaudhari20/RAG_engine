# rag_engine/ingestion/text_normalizer.py
import re

def normalize_text(text: str) -> str:
    # Fix missing spaces between words (very basic)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
