# rag_engine/ingestion/paper_parser.py
import uuid
from typing import List

from rag_engine.schema.block import Block
from rag_engine.schema.block_type import BlockType
from rag_engine.schema.document import Document

from rag_engine.ingestion.pdf_loader import load_pdf_text
from rag_engine.ingestion.section_detector import detect_sections
from rag_engine.ingestion.text_normalizer import normalize_text


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def parse_paper(pdf_path: str, doc_id: str = None) -> Document:
    raw_text = load_pdf_text(pdf_path)
    sections = detect_sections(raw_text)

    blocks: List[Block] = []
    order = 0

    for section_title, section_text in sections:
        paragraphs = split_paragraphs(section_text)

        for para in paragraphs:
            para = normalize_text(para)  # ✅ FIX: normalize inside loop

            block = Block(
                id=str(uuid.uuid4()),
                block_type=BlockType.PARAGRAPH,
                content=para,
                section_title=section_title,
                order_index=order
            )

            blocks.append(block)
            order += 1

    return Document(
        doc_id=doc_id or str(uuid.uuid4()),
        title="Parsed Research Paper",
        blocks=blocks
    )
