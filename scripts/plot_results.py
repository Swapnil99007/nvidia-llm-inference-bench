from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("results/raw/baseline_results.csv")
FIGURE_DIR = Path("results/figures")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    # Plot 1: latency by prompt
    plt.figure(figsize=(12, 6))
    plt.bar(df["prompt_id"], df["latency_sec"])
    plt.xlabel("Prompt ID")
    plt.ylabel("Latency (seconds)")
    plt.title("Latency by Prompt")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "latency_by_prompt.png")
    plt.close()

    # Plot 2: tokens/sec by prompt
    plt.figure(figsize=(12, 6))
    plt.bar(df["prompt_id"], df["tokens_per_sec"])
    plt.xlabel("Prompt ID")
    plt.ylabel("Tokens per second")
    plt.title("Tokens per Second by Prompt")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tokens_per_sec_by_prompt.png")
    plt.close()

    # Plot 3: average latency by category
    category_latency = df.groupby("category", as_index=False)["latency_sec"].mean()

    plt.figure(figsize=(10, 5))
    plt.bar(category_latency["category"], category_latency["latency_sec"])
    plt.xlabel("Category")
    plt.ylabel("Average Latency (seconds)")
    plt.title("Average Latency by Prompt Category")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "avg_latency_by_category.png")
    plt.close()

    print(f"Saved plots to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()