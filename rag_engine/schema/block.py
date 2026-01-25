# rag_engine/schema/block.py
from dataclasses import dataclass, field
from typing import Dict, Optional
from rag_engine.schema.block_type import BlockType


@dataclass
class Block:
    id: str
    block_type: BlockType
    content: str

    # Structural context
    section_title: Optional[str] = None
    section_number: Optional[str] = None

    # Order in document (important for context reconstruction)
    order_index: int = 0

    # Flexible metadata (table number, algorithm id, etc.)
    metadata: Dict = field(default_factory=dict)

    def preview(self, length: int = 200) -> str:
        """Short preview for debugging / UI."""
        return self.content[:length] + ("..." if len(self.content) > length else "")
