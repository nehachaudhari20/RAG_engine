# test.py
import os

from rag_engine.ingestion.paper_parser import parse_paper

def main():
    # Path to sample PDF
    pdf_path = os.path.join("data", "sample_paper.pdf")
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found at {pdf_path}")
        return

    # Parse the paper
    doc = parse_paper(pdf_path, doc_id="test_doc")

    # Display summary
    print("=" * 80)
    print(f"Parsed Document ID: {doc.doc_id}")
    print(f"Total blocks: {len(doc.blocks)}")
    print("=" * 80)

    # Show first few blocks
    for i, block in enumerate(doc.blocks[:10]):
        print(f"[{i:02d}] Section: {block.section_title}")
        print(f"    Type: {block.block_type}")
        print(f"    Preview: {block.preview(150)}")
        print("-" * 80)

if __name__ == "__main__":
    main()
