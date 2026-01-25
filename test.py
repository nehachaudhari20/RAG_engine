# test_chunking.py
from rag_engine.ingestion.paper_parser import parse_paper
from rag_engine.chunking.semantic_chunker import SemanticChunker
from rag_engine.embeddings.embedder import Embedder

doc = parse_paper("data/sample_paper.pdf")

embedder = Embedder()
chunker = SemanticChunker(embedder)

chunks = chunker.chunk(doc.blocks)

print(f"Total semantic chunks: {len(chunks)}")

for c in chunks[:5]:
    print(
        f"[Chunk] paragraphs={c.metadata['num_paragraphs']}, "
        f"section={c.section_title}"
    )
    print(c.preview(200))
    print("-" * 80)
