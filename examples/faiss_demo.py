from rag_engine.ingestion.loaders.text_loader import load_text
from rag_engine.chunking.overlap import OverlapChunker
from rag_engine.embeddings.local import LocalEmbeddingModel
from rag_engine.embeddings.utils import embed_chunks
from rag_engine.vector_store.faiss_store import FaissVectorStore

# 1. Load docs
docs = [
    load_text("Redis connection pool exhausted causing timeout"),
    load_text("Payment failed due to database connection error"),
    load_text("Sunny weather in Bangalore today")
]

# 2. Chunk
chunker = OverlapChunker(chunk_size=80, overlap=20)
chunks = []
for d in docs:
    chunks.extend(chunker.chunk(d))

# 3. Embed
embedder = LocalEmbeddingModel()
chunks = embed_chunks(chunks, embedder)

# 4. Index in FAISS
store = FaissVectorStore(dim=len(chunks[0].embedding))
store.add(chunks)

# 5. Query
query = "Why did the payment system timeout?"
q_vec = embedder.embed_query(query)

results = store.search(q_vec, top_k=3)

for chunk, score in results:
    print(round(score, 3), "→", chunk.content)
