from rag_engine.engine import RAGEngine
from rag_engine.ingestion.loaders.pdf_loader import load_pdf
from rag_engine.chunking.overlap import OverlapChunker
from rag_engine.embeddings.local import LocalEmbeddingModel
from rag_engine.vector_store.faiss_store import FaissVectorStore

# 1. Initialize engine
chunker = OverlapChunker(chunk_size=400, overlap=100)
embedder = LocalEmbeddingModel()
vector_store = FaissVectorStore(dim=384)

engine = RAGEngine(
    chunker=chunker,
    embedding_model=embedder,
    vector_store=vector_store,
    top_k=5
)

# 2. Load PDF
documents = load_pdf("sample.pdf")
print(f"Ingesting {len(documents)} pages")

engine.add_documents(documents)

# 3. Run real queries
queries = [
    "What is RCA?",
    "What is Hierarchical Learning?",
    "How is evaluation done?",
]

for q in queries:
    print(f"\nQUERY: {q}")
    response = engine.query(q)

    for r in response.results:
        page = r.metadata.get("page_number")
        snippet = r.content[:120].replace("\n", " ")

        print(
            f"Score={round(r.score, 3)} | Page={page} | {snippet}..."
        )
