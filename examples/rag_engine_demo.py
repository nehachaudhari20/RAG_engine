from rag_engine.engine import RAGEngine
from rag_engine.ingestion.loaders.text_loader import load_text
from rag_engine.chunking.overlap import OverlapChunker
from rag_engine.embeddings.local import LocalEmbeddingModel
from rag_engine.vector_store.faiss_store import FaissVectorStore

# Initialize components
chunker = OverlapChunker(chunk_size=80, overlap=20)
embedder = LocalEmbeddingModel()
vector_store = FaissVectorStore(dim=384)

engine = RAGEngine(
    chunker=chunker,
    embedding_model=embedder,
    vector_store=vector_store,
    top_k=3
)

# Ingest documents
docs = [
    load_text("Redis connection pool exhausted causing payment timeout"),
    load_text("Database connection error during transaction processing"),
    load_text("Weather is sunny in Bangalore today")
]

engine.add_documents(docs)

# Query
response = engine.query("Why did the payment system timeout?")

for r in response.results:
    print(round(r.score, 3), "→", r.content)
