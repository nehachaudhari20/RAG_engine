from rag_engine.ingestion.loaders.pdf_loader import load_pdf

docs = load_pdf("sample.pdf")

for d in docs:
    print(f"Page {d.metadata['page_number']} -> {len(d.content)} chars")
