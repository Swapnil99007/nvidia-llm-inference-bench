from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RAW_DIR = Path("results/raw")


def latest_run(pattern: str) -> Path:
    matches = list(RAW_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No runs found for pattern: {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def plot_metric(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[y_col], marker="o")
    plt.xlabel("Target QPS")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(df[x_col])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    run_dir = latest_run("phase5_trtllm_qps_*")
    summary_file = run_dir / "qps_summary.csv"

    figure_dir = Path("results/figures") / run_dir.name
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_file)

    plot_metric(
        df,
        "qps_target",
        "avg_latency_sec",
        "Average Latency (seconds)",
        "TensorRT-LLM Average Latency vs Target QPS",
        figure_dir / "avg_latency_vs_qps.png",
    )

    plot_metric(
        df,
        "qps_target",
        "p95_latency_sec",
        "P95 Latency (seconds)",
        "TensorRT-LLM P95 Latency vs Target QPS",
        figure_dir / "p95_latency_vs_qps.png",
    )

    plot_metric(
        df,
        "qps_target",
        "p99_latency_sec",
        "P99 Latency (seconds)",
        "TensorRT-LLM P99 Latency vs Target QPS",
        figure_dir / "p99_latency_vs_qps.png",
    )

    plot_metric(
        df,
        "qps_target",
        "achieved_requests_per_sec",
        "Achieved Requests/sec",
        "TensorRT-LLM Achieved Throughput vs Target QPS",
        figure_dir / "throughput_vs_qps.png",
    )

    plot_metric(
        df,
        "qps_target",
        "success_rate",
        "Success Rate",
        "TensorRT-LLM Success Rate vs Target QPS",
        figure_dir / "success_rate_vs_qps.png",
    )

    print(f"Read summary from: {summary_file}")
    print(f"Saved plots to: {figure_dir}")


if __name__ == "__main__":
    main()