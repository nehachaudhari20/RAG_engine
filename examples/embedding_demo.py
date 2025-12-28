from rag_engine.ingestion.loaders.text_loader import load_text
from rag_engine.chunking.overlap import OverlapChunker
from rag_engine.embeddings.local import LocalEmbeddingModel
from rag_engine.embeddings.utils import embed_chunks

doc = load_text("Redis connection pool exhausted causing timeout errors")
chunker = OverlapChunker(chunk_size=50, overlap=10)
chunks = chunker.chunk(doc)

embedder = LocalEmbeddingModel()
chunks = embed_chunks(chunks, embedder)

print(len(chunks))
print(len(chunks[0].embedding))  # should be 384
