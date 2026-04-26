import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from transformers import AutoTokenizer


MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
BENCHMARK_CONFIG_PATH = Path("configs/benchmark_matrix.yaml")
RESULTS_BASE_DIR = Path("results/raw")

PROMPTS_FILE_FALLBACK = Path("prompts/prompts.jsonl")

VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_API_KEY = "token-abc123"
ENGINE_NAME = "vllm"

QPS_LEVELS = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]
TEST_DURATION_SEC = 60
SETTING_LABEL = "default_output"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_vals[lower]
    weight = rank - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def get_setting_max_tokens(bench_cfg: dict[str, Any], setting_label: str) -> int:
    for setting in bench_cfg["settings"]:
        if setting["label"] == setting_label:
            return int(setting["max_new_tokens"])
    raise ValueError(f"Setting label not found: {setting_label}")


def safe_token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def generate_with_vllm(prompt_text: str, model_name: str, max_tokens: int) -> tuple[str, float, str | None]:
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

    start = time.perf_counter()
    response = requests.post(url, headers=headers, json=payload, timeout=300)
    end = time.perf_counter()

    response.raise_for_status()
    data = response.json()

    generated_text = data["choices"][0]["text"]
    finish_reason = data["choices"][0].get("finish_reason")
    latency_sec = end - start
    return generated_text, latency_sec, finish_reason


def run_single_request(
    request_index: int,
    qps_target: int,
    prompt_item: dict[str, Any],
    model_name: str,
    tokenizer,
    max_tokens: int,
    scheduled_send_time: float,
) -> dict[str, Any]:
    now = time.perf_counter()
    if scheduled_send_time > now:
        time.sleep(scheduled_send_time - now)

    actual_send_time = time.time()

    prompt_id = prompt_item["id"]
    category = prompt_item["category"]
    prompt_text = prompt_item["prompt"]

    try:
        generated_only_text, latency_sec, finish_reason = generate_with_vllm(
            prompt_text=prompt_text,
            model_name=model_name,
            max_tokens=max_tokens,
        )
        success = True
        error_message = None
        output_token_count = safe_token_count(tokenizer, generated_only_text)
    except Exception as e:
        generated_only_text = ""
        latency_sec = None
        finish_reason = None
        success = False
        error_message = str(e)
        output_token_count = 0

    completed_time = time.time()

    return {
        "engine_name": ENGINE_NAME,
        "qps_target": qps_target,
        "request_index": request_index,
        "prompt_id": prompt_id,
        "category": category,
        "send_timestamp_unix": actual_send_time,
        "response_timestamp_unix": completed_time,
        "latency_sec": latency_sec,
        "success": success,
        "error_message": error_message,
        "finish_reason": finish_reason,
        "input_token_count": safe_token_count(tokenizer, prompt_text),
        "output_token_count": output_token_count,
        "prompt_text": prompt_text,
        "generated_only_text": generated_only_text,
    }


def main() -> None:
    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    bench_cfg = load_yaml(BENCHMARK_CONFIG_PATH)

    model_name = model_cfg["model_name"]
    prompt_file = Path(bench_cfg.get("prompt_file", PROMPTS_FILE_FALLBACK))
    max_tokens = get_setting_max_tokens(bench_cfg, SETTING_LABEL)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = load_prompts(prompt_file)

    run_name = "phase5_vllm_qps"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE_DIR / f"{run_name}_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    for qps in QPS_LEVELS:
        total_requests = qps * TEST_DURATION_SEC
        print(f"\nRunning QPS level: {qps} for {TEST_DURATION_SEC}s ({total_requests} requests)")

        futures = []
        start_perf = time.perf_counter()

        max_workers = max(qps * 2, 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(total_requests):
                prompt_item = prompts[i % len(prompts)]
                scheduled_send_time = start_perf + (i / qps)

                futures.append(
                    executor.submit(
                        run_single_request,
                        i,
                        qps,
                        prompt_item,
                        model_name,
                        tokenizer,
                        max_tokens,
                        scheduled_send_time,
                    )
                )

            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)

                latency_display = (
                    f"{result['latency_sec']:.4f}s"
                    if result["latency_sec"] is not None
                    else "ERROR"
                )
                print(
                    f"[qps={qps} | req={result['request_index']} | {result['prompt_id']}] "
                    f"latency={latency_display} success={result['success']}"
                )

    request_df = pd.DataFrame(all_results)
    request_log_file = run_dir / "request_log.csv"
    request_df.to_csv(request_log_file, index=False)

    summary_rows = []
    for qps in QPS_LEVELS:
        subset = request_df[request_df["qps_target"] == qps]
        success_subset = subset[subset["success"] == True]

        latencies = success_subset["latency_sec"].dropna().tolist()
        total_requests = len(subset)
        completed_requests = len(success_subset)
        failed_requests = total_requests - completed_requests
        success_rate = completed_requests / total_requests if total_requests > 0 else 0.0

        if not subset.empty:
            wall_clock_sec = (
                subset["response_timestamp_unix"].max() - subset["send_timestamp_unix"].min()
            )
        else:
            wall_clock_sec = 0.0

        achieved_rps = completed_requests / wall_clock_sec if wall_clock_sec > 0 else 0.0
        avg_output_tokens = (
            float(success_subset["output_token_count"].mean()) if completed_requests > 0 else 0.0
        )
        avg_tokens_per_sec = (
            float(
                (success_subset["output_token_count"] / success_subset["latency_sec"]).mean()
            )
            if completed_requests > 0
            else 0.0
        )

        summary_rows.append(
            {
                "engine_name": ENGINE_NAME,
                "setting_label": SETTING_LABEL,
                "qps_target": qps,
                "test_duration_sec": TEST_DURATION_SEC,
                "total_requests": total_requests,
                "completed_requests": completed_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate,
                "achieved_requests_per_sec": achieved_rps,
                "avg_latency_sec": statistics.mean(latencies) if latencies else None,
                "p50_latency_sec": percentile(latencies, 0.50),
                "p95_latency_sec": percentile(latencies, 0.95),
                "p99_latency_sec": percentile(latencies, 0.99),
                "avg_output_tokens": avg_output_tokens,
                "avg_tokens_per_sec": avg_tokens_per_sec,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_file = run_dir / "qps_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    metadata = {
        "run_name": run_name,
        "run_timestamp": run_timestamp,
        "engine_name": ENGINE_NAME,
        "model_name": model_name,
        "prompt_file": str(prompt_file),
        "setting_label": SETTING_LABEL,
        "max_new_tokens": max_tokens,
        "qps_levels": QPS_LEVELS,
        "test_duration_sec": TEST_DURATION_SEC,
        "host": VLLM_HOST,
        "port": VLLM_PORT,
    }
    metadata_file = run_dir / "run_metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved request log to: {request_log_file}")
    print(f"Saved QPS summary to: {summary_file}")
    print(f"Saved metadata to: {metadata_file}")


if __name__ == "__main__":
    main()