import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer
from tensorrt_llm import LLM, SamplingParams


MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
BENCHMARK_CONFIG_PATH = Path("configs/benchmark_matrix.yaml")
RESULTS_BASE_DIR = Path("results/raw")

ENGINE_NAME = "tensorrt_llm"


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: Path):
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


def safe_token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_sampling_params(max_new_tokens: int):
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )


def main():
    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    bench_cfg = load_yaml(BENCHMARK_CONFIG_PATH)

    model_name = model_cfg["model_name"]
    prompt_file = Path(bench_cfg["prompt_file"])
    settings = bench_cfg["settings"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name)

    run_name = f"{bench_cfg['run_name']}_trtllm"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE_DIR / f"{run_name}_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompt_file)
    all_results = []

    for setting in settings:
        label = setting["label"]
        max_new_tokens = setting["max_new_tokens"]

        print(f"\nRunning TensorRT-LLM setting: {label}")
        sampling_params = build_sampling_params(max_new_tokens)

        for item in prompts:
            prompt_id = item["id"]
            category = item["category"]
            prompt_text = item["prompt"]

            start_time = time.perf_counter()
            outputs = llm.generate([prompt_text], sampling_params)
            end_time = time.perf_counter()

            output = outputs[0]
            generated_only_text = output.outputs[0].text if output.outputs else ""
            latency_sec = end_time - start_time

            input_token_count = safe_token_count(tokenizer, prompt_text)
            output_token_count = safe_token_count(tokenizer, generated_only_text)
            tokens_per_sec = (
                output_token_count / latency_sec if latency_sec > 0 else None
            )

            finish_reason = None
            if output.outputs and hasattr(output.outputs[0], "finish_reason"):
                finish_reason = output.outputs[0].finish_reason

            row = {
                "run_name": run_name,
                "run_timestamp": run_timestamp,
                "engine_name": ENGINE_NAME,
                "setting_label": label,
                "model_name": model_name,
                "device": "gpu",
                "prompt_id": prompt_id,
                "category": category,
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
                "max_new_tokens": max_new_tokens,
                "latency_sec": latency_sec,
                "tokens_per_sec": tokens_per_sec,
                "finish_reason": finish_reason,
                "prompt_text": prompt_text,
                "output_text": prompt_text + generated_only_text,
                "generated_only_text": generated_only_text,
            }
            all_results.append(row)

            tok_s_display = f"{tokens_per_sec:.2f}" if tokens_per_sec is not None else "NA"
            print(
                f"[{label} | {prompt_id}] "
                f"latency={latency_sec:.4f}s "
                f"tok/s={tok_s_display} "
                f"finish_reason={finish_reason}"
            )

    df = pd.DataFrame(all_results)

    results_file = run_dir / "benchmark_results.csv"
    df.to_csv(results_file, index=False)

    metadata = {
        "run_name": run_name,
        "run_timestamp": run_timestamp,
        "engine_name": ENGINE_NAME,
        "model_name": model_name,
        "prompt_file": str(prompt_file),
        "num_prompts": len(prompts),
        "num_settings": len(settings),
    }
    metadata_file = run_dir / "run_metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    summary = {
        "run_name": run_name,
        "run_timestamp": run_timestamp,
        "engine_name": ENGINE_NAME,
        "model_name": model_name,
        "num_prompts": len(prompts),
        "num_settings": len(settings),
        "avg_latency_sec": float(df["latency_sec"].mean()),
        "min_latency_sec": float(df["latency_sec"].min()),
        "max_latency_sec": float(df["latency_sec"].max()),
        "avg_tokens_per_sec": float(df["tokens_per_sec"].mean()),
        "min_tokens_per_sec": float(df["tokens_per_sec"].min()),
        "max_tokens_per_sec": float(df["tokens_per_sec"].max()),
    }
    summary_file = run_dir / "run_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to: {results_file}")
    print(f"Saved metadata to: {metadata_file}")
    print(f"Saved summary to: {summary_file}")


if __name__ == "__main__":
    main()