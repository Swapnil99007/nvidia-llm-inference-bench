from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RAW_DIR = Path("results/raw")


def latest_run(pattern: str) -> Path:
    matches = list(RAW_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No runs found for pattern: {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def main():
    run_dir = latest_run("phase31_vllm_concurrency_qwen25_7b_instruct_*")
    summary_file = run_dir / "concurrency_summary.csv"
    FIGURE_DIR = Path("results/figures") / run_dir.name

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_file)

    plt.figure(figsize=(8, 5))
    plt.plot(df["concurrency_level"], df["avg_latency_sec"], marker="o")
    plt.xlabel("Concurrency Level")
    plt.ylabel("Average Latency (seconds)")
    plt.title("vLLM Average Latency vs Concurrency")
    plt.xticks(df["concurrency_level"])
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "latency_vs_concurrency.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["concurrency_level"], df["avg_tokens_per_sec"], marker="o")
    plt.xlabel("Concurrency Level")
    plt.ylabel("Average Tokens/sec")
    plt.title("vLLM Average Tokens/sec vs Concurrency")
    plt.xticks(df["concurrency_level"])
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tokens_per_sec_vs_concurrency.png")
    plt.close()

    print(f"Saved plots to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()