# rag_engine/chunking/structure_rules.py
from rag_engine.schema.block_type import BlockType


def is_atomic(block_type: BlockType) -> bool:
    """
    Atomic blocks must never be split or merged.
    """
    return block_type in {
        BlockType.TABLE,
        BlockType.ALGORITHM
    }
