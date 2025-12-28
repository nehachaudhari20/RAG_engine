from rag_engine.ingestion.loaders.text_loader import load_text
from rag_engine.chunking.overlap import OverlapChunker

doc = load_text("A" * 1200)
chunker = OverlapChunker(chunk_size=500, overlap=100)

chunks = chunker.chunk(doc)
print(len(chunks))
