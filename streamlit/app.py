# streamlit/app.py

import streamlit as st
import tempfile
import os

from rag_engine.ingestion.paper_parser import parse_paper
from rag_engine.chunking.semantic_chunker import SemanticChunker
from rag_engine.embeddings.embedder import Embedder
from rag_engine.index.index_manager import IndexManager
from rag_engine.retrieval.retriever import Retriever
from ui_components import render_chunk


st.set_page_config(page_title="Research Paper RAG", layout="wide")

st.title("📄 Structure-Aware Research Paper RAG")

st.markdown(
    """
This system uses **semantic + structure-aware chunking** instead of
fixed token windows.
"""
)

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # -----------------------------
    # Build pipeline
    # -----------------------------
    with st.spinner("Parsing paper..."):
        doc = parse_paper(pdf_path)

    embedder = Embedder()
    chunker = SemanticChunker(embedder)

    with st.spinner("Semantic chunking..."):
        chunks = chunker.chunk(doc.blocks)

    index_manager = IndexManager(embedder)
    index_manager.build(chunks)

    retriever = Retriever(index_manager)

    st.success(f"Parsed {len(doc.blocks)} paragraphs → {len(chunks)} semantic chunks")

    # -----------------------------
    # Show chunks
    # -----------------------------
    with st.expander("🔍 View Semantic Chunks"):
        for c in chunks:
            render_chunk(c)

    # -----------------------------
    # Query
    # -----------------------------
    st.subheader("Ask a question")

    query = st.text_input("Enter a research question")

    if query:
        results = retriever.retrieve(query, top_k=5)

        st.subheader("Retrieved Context")
        for r in results:
            render_chunk(r)

    os.remove(pdf_path)
