# Structure-Aware RAG Engine for Research Papers

This project implements a **research-grade Retrieval-Augmented Generation (RAG) engine**
designed specifically for **research papers**.

Unlike standard RAG pipelines that rely on fixed-size or token-based chunking, this system
**explicitly understands document structure and semantics** before retrieval.

The core focus is **high-quality chunking and retrieval**, not prompt engineering or UI tricks.

---

## 🚀 Key Idea

> **Retrieval quality depends more on how documents are chunked than on how prompts are written.**

Most RAG systems treat documents as flat text and split them arbitrarily.
This project instead:
- parses research papers into structured elements
- groups paragraphs into **semantic chunks**
- retrieves context based on **meaning and structure**, not token windows

---

## ✨ Features

### 📘 Structure-Aware Ingestion
- Parses research papers (PDFs) into:
  - sections
  - paragraphs
- Preserves:
  - section titles
  - document order
- Normalizes noisy PDF text for better embeddings

### 🧩 Semantic Chunking (Core Contribution)
- Chunks are **not token-based**
- Adjacent paragraphs are grouped **only if they are semantically similar**
- Topic shifts trigger new chunks
- Designed to keep:
  - explanations
  - algorithms
  - conceptual blocks intact

### 🔍 Vector-Based Retrieval
- Uses **Sentence-Transformers** for embeddings
- Uses **FAISS** for vector indexing
- Retrieves top-k **semantic chunks**, not raw paragraphs
- Preserves document order for readable context

### 🎨 Streamlit Visualization
- Upload a research paper
- Inspect:
  - semantic chunks
  - retrieved context
- Acts as a **debugger for document intelligence**, not a chatbot

---

## 🏗️ Project Structure

```

rag_engine/
│
├── rag_engine/                # Core RAG engine (installable package)
│   ├── ingestion/             # PDF parsing & section detection
│   ├── schema/                # Block & document abstractions
│   ├── chunking/              # Semantic + structure-aware chunking
│   ├── embeddings/            # Embedding interface
│   ├── index/                 # FAISS vector index
│   ├── retrieval/             # Retrieval logic
│   └── pipeline.py            # End-to-end orchestration
│
├── streamlit/                 # Visualization layer
│   ├── app.py
│   └── ui_components.py
│
├── evaluation/                # (Planned) RAG evaluation
├── data/                      # Sample PDFs
├── pyproject.toml
├── requirements.txt
└── README.md

````

---

## 🧠 Design Philosophy

- **Chunking > Prompting**
- **Structure before embeddings**
- **Simple, inspectable components**
- **Engine-level design**, not a chat app

This system is designed to be:
- reusable
- testable
- extendable
- consumable by agentic or reasoning systems

---

## 🧪 Example Workflow

1. Upload a research paper (PDF)
2. Parse it into structured blocks
3. Create semantic chunks based on topic coherence
4. Index chunks using FAISS
5. Retrieve relevant chunks for a query
6. Inspect results via Streamlit UI

---

## 🛠️ Installation

```bash
pip install -e .
````

Run the Streamlit app:

```bash
streamlit run streamlit/app.py
```

---

## 📌 Current Status

* ✅ Structure-aware ingestion
* ✅ Semantic chunking
* ✅ Vector retrieval
* ✅ Streamlit visualization
* ⏳ RAGAS-based evaluation (planned)

---

## 🔮 Future Work

* Quantitative evaluation using **RAGAS**
* Algorithm & table-specific chunk handling
* Intent-aware retrieval (definition vs comparison queries)
* LLM-based answer generation layer


