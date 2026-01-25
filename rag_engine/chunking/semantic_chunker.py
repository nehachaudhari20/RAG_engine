# rag_engine/chunking/semantic_chunker.py
import uuid
from typing import List

from rag_engine.schema.block import Block
from rag_engine.schema.block_type import BlockType
from rag_engine.chunking.base_chunker import BaseChunker
from rag_engine.chunking.semantic_splitter import should_split
from rag_engine.chunking.structure_rules import is_atomic


class SemanticChunker(BaseChunker):

    def __init__(self, embedder, similarity_threshold: float = 0.75):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def chunk(self, blocks: List[Block]) -> List[Block]:
        chunks: List[Block] = []

        current_chunk_blocks: List[Block] = []
        prev_embedding = None
        order = 0

        for block in blocks:
            # Only chunk paragraphs for now
            if block.block_type != BlockType.PARAGRAPH:
                continue

            curr_embedding = self.embedder.embed(block.content)

            if prev_embedding is None:
                current_chunk_blocks = [block]
            else:
                if should_split(prev_embedding, curr_embedding, self.similarity_threshold):
                    chunks.append(self._merge_blocks(current_chunk_blocks, order))
                    order += 1
                    current_chunk_blocks = [block]
                else:
                    current_chunk_blocks.append(block)

            prev_embedding = curr_embedding

        if current_chunk_blocks:
            chunks.append(self._merge_blocks(current_chunk_blocks, order))

        return chunks

    def _merge_blocks(self, blocks: List[Block], order_index: int) -> Block:
        content = "\n\n".join(b.content for b in blocks)

        return Block(
            id=str(uuid.uuid4()),
            block_type=BlockType.PARAGRAPH,
            content=content,
            section_title=blocks[0].section_title,
            order_index=order_index,
            metadata={
                "num_paragraphs": len(blocks),
                "source_block_ids": [b.id for b in blocks]
            }
        )
