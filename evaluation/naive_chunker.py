# evaluation/naive_chunker.py
from typing import List
from rag_engine.schema.block import Block
from rag_engine.schema.block_type import BlockType


def naive_chunk(blocks: List[Block]) -> List[Block]:
    """
    Baseline: treat each paragraph as its own chunk.
    """
    return [
        b for b in blocks
        if b.block_type == BlockType.PARAGRAPH
    ]
