# src/rag_engine/ingestion/loaders/pdf_loader.py

from typing import List
from pathlib import Path
from pypdf import PdfReader

from rag_engine.ingestion.normalizer import normalize_document
from rag_engine.schema.document import Document


def load_pdf(path: str) -> List[Document]:
    """
    Load a PDF file and convert each page into a Document.
    Each page is treated as a separate document for better traceability.
    """
    pdf_path = Path(path)
    reader = PdfReader(pdf_path)

    documents: List[Document] = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        doc = normalize_document(
            content=text,
            source="pdf",
            metadata={
                "file_name": pdf_path.name,
                "page_number": page_num
            }
        )
        documents.append(doc)

    return documents
