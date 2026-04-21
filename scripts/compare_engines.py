from pathlib import Path
import pandas as pd


RAW_DIR = Path("results/raw")

ENGINE_PATTERNS = {
    "hf_transformers": "phase3_engine_comparison_qwen25_7b_instruct_[0-9]*",
    "vllm": "phase3_engine_comparison_qwen25_7b_instruct_vllm_[0-9]*",
    "tensorrt_llm": "phase3_engine_comparison_qwen25_7b_instruct_trtllm_[0-9]*",
}


def latest_run(pattern: str) -> Path:
    matches = list(RAW_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No runs found for pattern: {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def main():
    all_dfs = []
    chosen_runs = {}

    for engine_name, pattern in ENGINE_PATTERNS.items():
        run_dir = latest_run(pattern)
        csv_file = run_dir / "benchmark_results.csv"
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
        chosen_runs[engine_name] = run_dir

    combined = pd.concat(all_dfs, ignore_index=True)

    summary = (
        combined.groupby(["engine_name", "setting_label", "category"], as_index=False)
        .agg(
            avg_latency_sec=("latency_sec", "mean"),
            avg_tokens_per_sec=("tokens_per_sec", "mean"),
            avg_input_tokens=("input_token_count", "mean"),
            avg_output_tokens=("output_token_count", "mean"),
            num_prompts=("prompt_id", "count"),
        )
    )

    output_file = RAW_DIR / "latest_engine_comparison_summary.csv"
    summary.to_csv(output_file, index=False)

    for engine_name, run_dir in chosen_runs.items():
        print(f"{engine_name}: {run_dir}")

    print(f"Saved comparison summary to: {output_file}")
    print(summary)


if __name__ == "__main__":
    main()