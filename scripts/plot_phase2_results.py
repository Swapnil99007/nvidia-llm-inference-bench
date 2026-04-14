from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

RUN_DIR = max(Path("results/raw").glob("phase2_baseline_distilgpt2_*"), key=lambda p: p.stat().st_mtime)
RESULTS_FILE = RUN_DIR / "benchmark_results.csv"
FIGURE_DIR = Path("results/figures") / RUN_DIR.name

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_FILE)

    plt.figure(figsize=(12, 6))
    plt.bar(df["prompt_id"], df["latency_sec"])
    plt.xlabel("Prompt ID")
    plt.ylabel("Latency (seconds)")
    plt.title("Phase 2 Latency by Prompt")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "latency_by_prompt.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(df["prompt_id"], df["tokens_per_sec"])
    plt.xlabel("Prompt ID")
    plt.ylabel("Tokens per Second")
    plt.title("Phase 2 Tokens/sec by Prompt")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tokens_per_sec_by_prompt.png")
    plt.close()

    category_latency = df.groupby("category", as_index=False)["latency_sec"].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(category_latency["category"], category_latency["latency_sec"])
    plt.xlabel("Category")
    plt.ylabel("Average Latency (seconds)")
    plt.title("Phase 2 Average Latency by Category")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "avg_latency_by_category.png")
    plt.close()

    category_tps = df.groupby("category", as_index=False)["tokens_per_sec"].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(category_tps["category"], category_tps["tokens_per_sec"])
    plt.xlabel("Category")
    plt.ylabel("Average Tokens/sec")
    plt.title("Phase 2 Average Tokens/sec by Category")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "avg_tokens_per_sec_by_category.png")
    plt.close()

    print(f"Saved plots to: {FIGURE_DIR}")

if __name__ == "__main__":
    main()