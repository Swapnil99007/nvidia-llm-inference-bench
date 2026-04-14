from pathlib import Path
import pandas as pd

RUN_DIR = max(Path("results/raw").glob("phase2_baseline_distilgpt2_*"), key=lambda p: p.stat().st_mtime)

def main():
    input_file = RUN_DIR / "benchmark_results.csv"
    output_file = RUN_DIR / "summary_by_setting_and_category.csv"

    df = pd.read_csv(input_file)

    summary = (
        df.groupby(["setting_label", "category"], as_index=False)
        .agg(
            avg_latency_sec=("latency_sec", "mean"),
            min_latency_sec=("latency_sec", "min"),
            max_latency_sec=("latency_sec", "max"),
            avg_tokens_per_sec=("tokens_per_sec", "mean"),
            avg_input_tokens=("input_token_count", "mean"),
            avg_output_tokens=("output_token_count", "mean"),
            num_prompts=("prompt_id", "count"),
        )
    )

    summary.to_csv(output_file, index=False)
    print(f"Saved summary to: {output_file}")
    print(summary)

if __name__ == "__main__":
    main()