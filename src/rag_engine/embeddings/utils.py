from typing import List
from rag_engine.schema.document import Chunk
from rag_engine.embeddings.base import BaseEmbeddingModel


def embed_chunks(
    chunks: List[Chunk],
    embedding_model: BaseEmbeddingModel
) -> List[Chunk]:
    texts = [c.content for c in chunks]
    vectors = embedding_model.embed_documents(texts)

    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec

    return chunks
