# test_retrieval.py
from rag_engine.ingestion.paper_parser import parse_paper
from rag_engine.chunking.semantic_chunker import SemanticChunker
from rag_engine.embeddings.embedder import Embedder
from rag_engine.index.index_manager import IndexManager
from rag_engine.retrieval.retriever import Retriever

# 1. Parse paper
doc = parse_paper("data/sample_paper.pdf")

# 2. Chunk semantically
embedder = Embedder()
chunker = SemanticChunker(embedder)
chunks = chunker.chunk(doc.blocks)

print(f"Chunks indexed: {len(chunks)}")

# 3. Build index
index_manager = IndexManager(embedder)
index_manager.build(chunks)

# 4. Retrieve
retriever = Retriever(index_manager)

query = "What is self-attention and why is it useful?"
results = retriever.retrieve(query, top_k=5)

print("\nRetrieved chunks:\n")
for r in results:
    print(f"[Section: {r.section_title}]")
    print(r.preview(300))
    print("-" * 80)
