# rag_engine/rag/prompt_builder.py
from typing import List
from rag_engine.schema.block import Block


def build_prompt(question: str, contexts):
    context_text = "\n\n".join(
        f"[Section: {c.section_title}]\n{c.content}"
        for c in contexts
    )

    return f"""
You are answering a technical question using only the provided context.

Context:
{context_text}

Question:
{question}

Answer using only the information in the context above.
Explain the reasoning clearly and explicitly.
If the answer is not present, say you don't know.
""".strip()

