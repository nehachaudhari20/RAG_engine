# evaluation/ragas_eval.py
import json
import os
from dotenv import load_dotenv

load_dotenv()
from ragas import evaluate
from ragas.metrics import context_recall, faithfulness
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from rag_engine.ingestion.paper_parser import parse_paper
from rag_engine.chunking.semantic_chunker import SemanticChunker
from rag_engine.embeddings.embedder import Embedder
from rag_engine.index.index_manager import IndexManager
from rag_engine.retrieval.retriever import Retriever
from evaluation.naive_chunker import naive_chunk

def build_dataset(questions, retriever):
    rows = []
    for q in questions:
        contexts = retriever.retrieve(q["question"], top_k=5)
        rows.append({
            "question": q["question"],
            "contexts": [c.content for c in contexts],
            "ground_truth": q["ground_truth"],
            "answer": q["ground_truth"]  # oracle answer for eval
        })
    return Dataset.from_list(rows)


def run_eval(pdf_path: str):
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0,
        api_key=os.getenv("GOOGLE_API_KEY")
    )   

    judge_llm = LangchainLLMWrapper(gemini_llm)
    with open("evaluation/eval_dataset.json") as f:
        questions = json.load(f)

    # Parse paper
    doc = parse_paper(pdf_path)

    embedder = Embedder()

    # -------- Baseline --------
    naive_chunks = naive_chunk(doc.blocks)
    naive_index = IndexManager(embedder)
    naive_index.build(naive_chunks)
    naive_retriever = Retriever(naive_index)

    naive_ds = build_dataset(questions, naive_retriever)

    naive_result = evaluate(
        naive_ds,
        metrics=[context_recall, faithfulness],
        llm=judge_llm
    )

    # -------- Semantic --------
    chunker = SemanticChunker(embedder)
    semantic_chunks = chunker.chunk(doc.blocks)

    semantic_index = IndexManager(embedder)
    semantic_index.build(semantic_chunks)
    semantic_retriever = Retriever(semantic_index)

    semantic_ds = build_dataset(questions, semantic_retriever)

    semantic_result = evaluate(
        semantic_ds,
        metrics=[context_recall, faithfulness],
        llm=judge_llm
    )

    return naive_result, semantic_result
