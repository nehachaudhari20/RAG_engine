# rag_engine/schema/document.py
from dataclasses import dataclass
from typing import List
from rag_engine.schema.block import Block
from rag_engine.schema.block_type import BlockType


@dataclass
class Document:
    doc_id: str
    title: str
    blocks: List[Block]

    def get_blocks_by_type(self, block_type: BlockType) -> List[Block]:
        return [b for b in self.blocks if b.block_type == block_type]

    def sort_blocks(self):
        self.blocks.sort(key=lambda b: b.order_index)
