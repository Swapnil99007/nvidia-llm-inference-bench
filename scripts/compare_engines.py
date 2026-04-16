from pathlib import Path
import pandas as pd


RAW_DIR = Path("results/raw")


def latest_run(pattern: str) -> Path:
    matches = list(RAW_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No runs found for pattern: {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def main():
    hf_run = latest_run("phase3_engine_comparison_qwen25_7b_instruct_[0-9]*")
    vllm_run = latest_run("phase3_engine_comparison_qwen25_7b_instruct_vllm_[0-9]*")

    hf_file = hf_run / "benchmark_results.csv"
    vllm_file = vllm_run / "benchmark_results.csv"

    hf_df = pd.read_csv(hf_file)
    vllm_df = pd.read_csv(vllm_file)

    combined = pd.concat([hf_df, vllm_df], ignore_index=True)

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

    print(f"HF run: {hf_run}")
    print(f"vLLM run: {vllm_run}")
    print(f"Saved comparison summary to: {output_file}")
    print(summary)


if __name__ == "__main__":
    main()