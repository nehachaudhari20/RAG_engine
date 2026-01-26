from evaluation.ragas_eval import run_eval
from evaluation.metrics import print_results

naive, semantic = run_eval("data/sample_paper.pdf")

print_results("Naive Chunking", naive)
print_results("Semantic Chunking", semantic)
