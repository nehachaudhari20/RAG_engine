# src/rag_engine/engine.py

from typing import List

from rag_engine.schema.document import Document, Chunk
from rag_engine.schema.retrieval import RetrievalResponse
from rag_engine.chunking.base import BaseChunker
from rag_engine.embeddings.base import BaseEmbeddingModel
from rag_engine.embeddings.utils import embed_chunks
from rag_engine.vector_store.base import BaseVectorStore
from rag_engine.retrieval.semantic import SemanticRetriever


class RAGEngine:
    """
    Core RAG engine that orchestrates ingestion, embedding, indexing, and retrieval.
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        top_k: int = 5
    ):
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.retriever = SemanticRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            top_k=top_k
        )

    # ----------------------------
    # INGESTION PIPELINE
    # ----------------------------
    def add_documents(self, documents: List[Document]) -> None:
        """
        Ingest documents into the RAG engine.
        """
        all_chunks: List[Chunk] = []

        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        # Embed chunks
        all_chunks = embed_chunks(all_chunks, self.embedding_model)

        # Index chunks
        self.vector_store.add(all_chunks)

    # ----------------------------
    # RETRIEVAL PIPELINE
    # ----------------------------
    def query(self, query: str) -> RetrievalResponse:
        """
        Retrieve relevant chunks for a query.
        """
        return self.retriever.retrieve(query)
