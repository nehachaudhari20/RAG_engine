# evaluation/metrics.py
def print_results(name, result):
    print(f"\n{name}")
    print("-" * 40)
    for k, v in result.items():
        print(f"{k}: {v:.4f}")
