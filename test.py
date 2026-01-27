# test_answer.py
from rag_engine.pipeline import answer_question

result = answer_question(
    pdf_path="data/sample_paper.pdf",
    question="Why is self-attention more parallelizable than RNNs?"
)

print("\nANSWER:\n")
print(result["answer"])

print("\nCONTEXT USED:\n")
for c in result["contexts"]:
    print(f"- {c.section_title}")
