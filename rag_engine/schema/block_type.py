# rag_engine/schema/block_type.py
from enum import Enum


class BlockType(str, Enum):
    SECTION = "section"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    ALGORITHM = "algorithm"
