from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

RAW_DIR = Path("results/raw")
FIGURE_DIR = Path("results/figures/phase5d_a100_vllm_vs_triton_bs128_inflight")

ENGINE_PATTERNS = {
    "vllm_a100": "phase5_vllm_qps_20260427_043945",
    "triton_trtllm_bs128_inflight": "phase5_triton_qps_20260429_195009",
}

def latest_run(pattern: str) -> Path:
    matches = list(RAW_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No runs found for pattern: {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)

def plot_comparison(df, y_col, ylabel, title, output_name):
    plt.figure(figsize=(9, 5))
    for engine in df["engine_name"].unique():
        engine_df = df[df["engine_name"] == engine].sort_values("qps_target")
        plt.plot(engine_df["qps_target"], engine_df[y_col], marker="o", label=engine)
    plt.xlabel("Target QPS")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(df["qps_target"].unique()))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / output_name, dpi=200)
    plt.close()

def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    dfs, chosen_runs = [], {}

    for engine, pattern in ENGINE_PATTERNS.items():
        run_dir = latest_run(pattern)
        df = pd.read_csv(run_dir / "qps_summary.csv")
        df["engine_name"] = engine
        dfs.append(df)
        chosen_runs[engine] = str(run_dir)

    combined = pd.concat(dfs, ignore_index=True)
    output_file = RAW_DIR / "latest_phase5d_a100_vllm_vs_triton_bs128_inflight.csv"
    combined.to_csv(output_file, index=False)

    plots = [
        ("avg_latency_sec", "Average Latency (seconds)", "Average Latency vs QPS", "avg_latency_comparison_vs_qps.png"),
        ("p95_latency_sec", "P95 Latency (seconds)", "P95 Latency vs QPS", "p95_latency_comparison_vs_qps.png"),
        ("p99_latency_sec", "P99 Latency (seconds)", "P99 Latency vs QPS", "p99_latency_comparison_vs_qps.png"),
        ("achieved_requests_per_sec", "Achieved Requests/sec", "Achieved Throughput vs QPS", "throughput_comparison_vs_qps.png"),
        ("avg_tokens_per_sec", "Average Tokens/sec per Request", "Token Throughput per Request vs QPS", "token_throughput_comparison_vs_qps.png"),
        ("success_rate", "Success Rate", "Success Rate vs QPS", "success_rate_comparison_vs_qps.png"),
    ]

    for args in plots:
        plot_comparison(combined, *args)

    print("Chosen runs:")
    for engine, run_dir in chosen_runs.items():
        print(f"{engine}: {run_dir}")

    print(f"\nSaved combined summary to: {output_file}")
    print(f"Saved comparison plots to: {FIGURE_DIR}")

if __name__ == "__main__":
    main()
