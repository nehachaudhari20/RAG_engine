# RAG Engine

A **domain-agnostic, engine-level Retrieval-Augmented Generation (RAG) core**
designed for **evidence retrieval**, not answer generation.

This repository implements the **retrieval and memory substrate** for agentic
systems, reliability pipelines, and knowledge-grounded applications.

---

## ✨ Key Features

- 📄 Multi-source ingestion (text, JSON, logs, PDFs)
- ✂️ Pluggable chunking strategies (fixed, overlapping)
- 🧠 Real semantic embeddings (Sentence Transformers)
- 🔍 FAISS-based vector search (cosine similarity)
- 📦 Clean, structured retrieval outputs
- 🔌 Engine-grade API (`add_documents`, `query`)
- 🧱 No LLM dependency

---

## 🧠 What This Repo Is (and Is Not)

### ✅ This repo **IS**
- A **retrieval engine**
- A **memory layer** for agentic systems
- A reusable infra component
- Deterministic and testable

---

## 🏗 Architecture Overview

```

Documents (text / PDF / logs)
↓
Ingestion
↓
Chunking
↓
Embeddings
↓
FAISS Index
↓
Semantic Retrieval
↓
Structured Evidence Results

````

The engine exposes a **single public surface**:

```python
engine.add_documents(...)
engine.query(...)
````

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or install in editable mode (recommended):

```bash
pip install -e .
```

---

### 2. Minimal usage example

```python
from rag_engine.engine import RAGEngine
from rag_engine.ingestion.loaders.text_loader import load_text
from rag_engine.chunking.overlap import OverlapChunker
from rag_engine.embeddings.local import LocalEmbeddingModel
from rag_engine.vector_store.faiss_store import FaissVectorStore

engine = RAGEngine(
    chunker=OverlapChunker(chunk_size=400, overlap=100),
    embedding_model=LocalEmbeddingModel(),
    vector_store=FaissVectorStore(dim=384),
    top_k=3
)

engine.add_documents([
    load_text("Redis connection pool exhausted causing timeout"),
    load_text("Database connection error during transaction")
])

response = engine.query("Why did the payment system timeout?")

for r in response.results:
    print(r.score, r.content)
```

---

## 📄 PDF Support

PDFs are treated as **first-class data sources**.

Each page is ingested as an independent document with metadata:

* `file_name`
* `page_number`

Example usage is available in:

```
examples/pdf_retrieval_demo.py
```

---

## 🔗 Intended Usage

This engine is designed to be consumed by:

* Agent runtimes
* Decision graphs
* RCA pipelines
* Distributed agentic RAG systems

For a full application using this engine, see:

> **`distributed-agentic-rag`** (separate repository)

---

## 🧪 Testing & Validation

This repo focuses on:

* correctness of retrieval
* semantic relevance
* deterministic behavior

Answer quality evaluation (e.g. RAGAS) is intentionally **out of scope**.

---

## 📌 Design Philosophy

* Engines should be reusable
* Retrieval should be deterministic
* Agents should not own memory
* Evaluation belongs to applications
* LLMs are optional, not foundational

---