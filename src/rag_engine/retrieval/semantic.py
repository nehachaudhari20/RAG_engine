# src/rag_engine/retrieval/semantic.py

from typing import List

from rag_engine.embeddings.base import BaseEmbeddingModel
from rag_engine.vector_store.base import BaseVectorStore
from rag_engine.schema.retrieval import RetrievalResult, RetrievalResponse


class SemanticRetriever:
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        top_k: int = 5
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> RetrievalResponse:
        # 1. Embed query
        query_vector = self.embedding_model.embed_query(query)

        # 2. Search vector store
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=self.top_k
        )

        # 3. Convert to structured response
        retrieval_results: List[RetrievalResult] = []

        for chunk, score in results:
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    score=score,
                    source=chunk.metadata.get("source", "unknown"),
                    metadata=chunk.metadata
                )
            )

        return RetrievalResponse(
            query=query,
            results=retrieval_results
        )
