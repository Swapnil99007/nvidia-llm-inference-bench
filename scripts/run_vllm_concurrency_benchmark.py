import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml
from transformers import AutoTokenizer


MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
RESULTS_BASE_DIR = Path("results/raw")
PROMPTS_FILE = Path("prompts/prompts.jsonl")

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_API_KEY = "token-abc123"
ENGINE_NAME = "vllm"

CONCURRENCY_LEVELS = [1, 2, 4, 8]
REQUESTS_PER_LEVEL = 16
MAX_TOKENS = 64


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: Path):
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


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
    finish_reason = data["choices"][0].get("finish_reason")
    latency_sec = end_time - start_time

    return generated_text, latency_sec, finish_reason


def run_single_request(prompt_item, model_name, tokenizer, concurrency_level, request_index):
    prompt_id = prompt_item["id"]
    category = prompt_item["category"]
    prompt_text = prompt_item["prompt"]

    generated_only_text, latency_sec, finish_reason = generate_with_vllm(
        prompt_text=prompt_text,
        model_name=model_name,
        max_tokens=MAX_TOKENS,
    )

    input_token_count = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    output_token_count = len(tokenizer(generated_only_text, add_special_tokens=False)["input_ids"])
    tokens_per_sec = output_token_count / latency_sec if latency_sec > 0 else None

    return {
        "engine_name": ENGINE_NAME,
        "prompt_id": prompt_id,
        "category": category,
        "concurrency_level": concurrency_level,
        "request_index": request_index,
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
        "latency_sec": latency_sec,
        "tokens_per_sec": tokens_per_sec,
        "finish_reason": finish_reason,
        "prompt_text": prompt_text,
        "generated_only_text": generated_only_text,
    }


def main():
    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    model_name = model_cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    run_name = "phase31_vllm_concurrency_qwen25_7b_instruct"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE_DIR / f"{run_name}_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(PROMPTS_FILE)
    prompts = prompts[:REQUESTS_PER_LEVEL]

    all_results = []

    for concurrency_level in CONCURRENCY_LEVELS:
        print(f"\nRunning concurrency level: {concurrency_level}")

        with ThreadPoolExecutor(max_workers=concurrency_level) as executor:
            futures = []

            for i, prompt_item in enumerate(prompts):
                futures.append(
                    executor.submit(
                        run_single_request,
                        prompt_item,
                        model_name,
                        tokenizer,
                        concurrency_level,
                        i,
                    )
                )

            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                print(
                    f"[concurrency={concurrency_level} | {result['prompt_id']}] "
                    f"latency={result['latency_sec']:.4f}s "
                    f"tok/s={result['tokens_per_sec']:.2f} "
                    f"finish_reason={result['finish_reason']}"
                )

    df = pd.DataFrame(all_results)
    results_file = run_dir / "benchmark_results.csv"
    df.to_csv(results_file, index=False)

    summary = (
        df.groupby(["concurrency_level"], as_index=False)
        .agg(
            avg_latency_sec=("latency_sec", "mean"),
            min_latency_sec=("latency_sec", "min"),
            max_latency_sec=("latency_sec", "max"),
            avg_tokens_per_sec=("tokens_per_sec", "mean"),
            avg_output_tokens=("output_token_count", "mean"),
            num_requests=("prompt_id", "count"),
        )
    )
    summary_file = run_dir / "concurrency_summary.csv"
    summary.to_csv(summary_file, index=False)

    metadata = {
        "run_name": run_name,
        "run_timestamp": run_timestamp,
        "engine_name": ENGINE_NAME,
        "model_name": model_name,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "requests_per_level": REQUESTS_PER_LEVEL,
        "max_tokens": MAX_TOKENS,
    }
    metadata_file = run_dir / "run_metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved results to: {results_file}")
    print(f"Saved summary to: {summary_file}")
    print(f"Saved metadata to: {metadata_file}")


if __name__ == "__main__":
    main()