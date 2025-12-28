from rag_engine.ingestion.loaders.text_loader import load_text
from rag_engine.chunking.overlap import OverlapChunker
from rag_engine.embeddings.local import LocalEmbeddingModel
from rag_engine.embeddings.utils import embed_chunks
from rag_engine.vector_store.faiss_store import FaissVectorStore
from rag_engine.retrieval.semantic import SemanticRetriever

# Load documents
docs = [
    load_text("Redis connection pool exhausted causing timeout"),
    load_text("Payment failed due to database connection error"),
    load_text("Sunny weather in Bangalore today")
]

# Chunk
chunker = OverlapChunker(chunk_size=80, overlap=20)
chunks = []
for d in docs:
    chunks.extend(chunker.chunk(d))

# Embed
embedder = LocalEmbeddingModel()
chunks = embed_chunks(chunks, embedder)

# Index
store = FaissVectorStore(dim=len(chunks[0].embedding))
store.add(chunks)

# Retrieve
retriever = SemanticRetriever(
    embedding_model=embedder,
    vector_store=store,
    top_k=2
)

response = retriever.retrieve("Why did the payment system timeout?")

for r in response.results:
    print(round(r.score, 3), "→", r.content)
