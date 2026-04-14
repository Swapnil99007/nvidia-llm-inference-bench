import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml


MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
BENCHMARK_CONFIG_PATH = Path("configs/benchmark_matrix.yaml")
RESULTS_BASE_DIR = Path("results/raw")

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_API_KEY = "token-abc123"
ENGINE_NAME = "vllm"


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: Path):
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def generate_with_vllm(prompt_text: str, model_name: str, max_tokens: int):
    url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/completions"
    headers = {
        "Authorization": f"Bearer {VLLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    start_time = time.perf_counter()
    response = requests.post(url, headers=headers, json=payload, timeout=300)
    end_time = time.perf_counter()

    response.raise_for_status()
    data = response.json()

    generated_text = data["choices"][0]["text"]
    latency_sec = end_time - start_time

    return generated_text, latency_sec


def main():
    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    bench_cfg = load_yaml(BENCHMARK_CONFIG_PATH)

    run_name = f"{bench_cfg['run_name']}_vllm"
    prompt_file = Path(bench_cfg["prompt_file"])
    settings = bench_cfg["settings"]
    model_name = model_cfg["model_name"]

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE_DIR / f"{run_name}_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompt_file)
    all_results = []

    for setting in settings:
        label = setting["label"]
        max_new_tokens = setting["max_new_tokens"]

        print(f"\nRunning vLLM setting: {label}")

        for item in prompts:
            prompt_id = item["id"]
            category = item["category"]
            prompt_text = item["prompt"]

            generated_only_text, latency_sec = generate_with_vllm(
                prompt_text=prompt_text,
                model_name=model_name,
                max_tokens=max_new_tokens,
            )

            input_token_count = estimate_token_count(prompt_text)
            output_token_count = estimate_token_count(generated_only_text)
            tokens_per_sec = output_token_count / latency_sec if latency_sec > 0 else None

            row = {
                "run_name": run_name,
                "run_timestamp": run_timestamp,
                "engine_name": ENGINE_NAME,
                "setting_label": label,
                "model_name": model_name,
                "device": "server_managed",
                "prompt_id": prompt_id,
                "category": category,
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
                "max_new_tokens": max_new_tokens,
                "latency_sec": latency_sec,
                "tokens_per_sec": tokens_per_sec,
                "prompt_text": prompt_text,
                "output_text": prompt_text + generated_only_text,
                "generated_only_text": generated_only_text,
            }
            all_results.append(row)

            print(
                f"[{label} | {prompt_id}] "
                f"latency={latency_sec:.4f}s "
                f"tok/s={tokens_per_sec:.2f}"
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
        "host": VLLM_HOST,
        "port": VLLM_PORT,
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