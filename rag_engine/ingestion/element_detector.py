# rag_engine/ingestion/element_detector.py
import re
from typing import List, Tuple


ALGORITHM_REGEX = re.compile(r"(Algorithm\s+\d+.*)", re.IGNORECASE)
TABLE_REGEX = re.compile(r"(Table\s+\d+.*)", re.IGNORECASE)


def detect_algorithms(text: str) -> List[str]:
    return ALGORITHM_REGEX.findall(text)


def detect_tables(text: str) -> List[str]:
    return TABLE_REGEX.findall(text)
