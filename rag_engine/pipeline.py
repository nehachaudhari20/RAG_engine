# rag_engine/pipeline.py
from rag_engine.ingestion.paper_parser import parse_paper
from rag_engine.chunking.semantic_chunker import SemanticChunker
from rag_engine.embeddings.embedder import Embedder
from rag_engine.index.index_manager import IndexManager
from rag_engine.retrieval.retriever import Retriever
from rag_engine.rag.answer_generator import AnswerGenerator


def answer_question(pdf_path: str, question: str, top_k: int = 5):
    # 1. Parse
    doc = parse_paper(pdf_path)

    # 2. Chunk
    embedder = Embedder()
    chunker = SemanticChunker(embedder)
    chunks = chunker.chunk(doc.blocks)

    # 3. Index
    index_manager = IndexManager(embedder)
    index_manager.build(chunks)

    # 4. Retrieve
    retriever = Retriever(index_manager)
    contexts = retriever.retrieve(question, top_k=top_k)

    # 5. Generate answer
    generator = AnswerGenerator()
    answer = generator.generate(question, contexts)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }
