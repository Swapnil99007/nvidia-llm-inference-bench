import json
import math
import queue
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tritonclient.grpc as grpcclient
import yaml
from transformers import AutoTokenizer


MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
BENCHMARK_CONFIG_PATH = Path("configs/benchmark_matrix.yaml")
RESULTS_BASE_DIR = Path("results/raw")

ENGINE_NAME = "triton_trtllm"
TRITON_URL = "localhost:8001"
MODEL_NAME = "ensemble"

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
            if line:
                prompts.append(json.loads(line))
    return prompts


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    values = sorted(values)
    rank = (len(values) - 1) * p
    lower = math.floor(rank)
    upper = math.ceil(rank)

    if lower == upper:
        return values[lower]

    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def get_setting_max_tokens(bench_cfg: dict[str, Any], setting_label: str) -> int:
    for setting in bench_cfg["settings"]:
        if setting["label"] == setting_label:
            return int(setting["max_new_tokens"])
    raise ValueError(f"Setting label not found: {setting_label}")


def safe_token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_inputs(prompt_text: str, max_tokens: int):
    text_input = np.array([[prompt_text]], dtype=object)
    max_tokens_arr = np.array([[max_tokens]], dtype=np.int32)

    inputs = []

    inp = grpcclient.InferInput("text_input", text_input.shape, "BYTES")
    inp.set_data_from_numpy(text_input)
    inputs.append(inp)

    inp = grpcclient.InferInput("max_tokens", max_tokens_arr.shape, "INT32")
    inp.set_data_from_numpy(max_tokens_arr)
    inputs.append(inp)

    return inputs


def decode_output(output_arr) -> str:
    if output_arr is None or len(output_arr) == 0:
        return ""

    value = output_arr[0]

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        value = value.item()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def generate_with_triton_stream(prompt_text: str, max_tokens: int) -> tuple[str, float]:
    q = queue.Queue()

    def callback(result, error):
        q.put((result, error))

    client = grpcclient.InferenceServerClient(url=TRITON_URL)

    start = time.perf_counter()
    client.start_stream(callback=callback)

    client.async_stream_infer(
        model_name=MODEL_NAME,
        inputs=build_inputs(prompt_text, max_tokens),
    )

    result, error = q.get(timeout=300)
    client.stop_stream()
    end = time.perf_counter()

    if error:
        raise RuntimeError(error)

    output_arr = result.as_numpy("text_output")
    generated_only_text = decode_output(output_arr)

    return generated_only_text, end - start


def run_single_request(
    request_index: int,
    qps_target: int,
    prompt_item: dict[str, Any],
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
        generated_only_text, latency_sec = generate_with_triton_stream(
            prompt_text=prompt_text,
            max_tokens=max_tokens,
        )

        success = True
        error_message = None
        finish_reason = None
        output_token_count = safe_token_count(tokenizer, generated_only_text)

    except Exception as e:
        generated_only_text = ""
        latency_sec = None
        success = False
        error_message = str(e)
        finish_reason = None
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

    hf_model_name = model_cfg["model_name"]
    prompt_file = Path(bench_cfg["prompt_file"])
    max_tokens = get_setting_max_tokens(bench_cfg, SETTING_LABEL)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    prompts = load_prompts(prompt_file)

    run_name = "phase5_triton_qps"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE_DIR / f"{run_name}_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    for qps in QPS_LEVELS:
        total_requests = qps * TEST_DURATION_SEC
        print(f"\nRunning Triton QPS level: {qps} for {TEST_DURATION_SEC}s ({total_requests} requests)")

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

        wall_clock_sec = (
            subset["response_timestamp_unix"].max() - subset["send_timestamp_unix"].min()
            if not subset.empty
            else 0.0
        )

        achieved_rps = completed_requests / wall_clock_sec if wall_clock_sec > 0 else 0.0

        avg_output_tokens = (
            float(success_subset["output_token_count"].mean())
            if completed_requests > 0
            else 0.0
        )

        avg_tokens_per_sec = (
            float((success_subset["output_token_count"] / success_subset["latency_sec"]).mean())
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
        "hf_model_name": hf_model_name,
        "triton_model_name": MODEL_NAME,
        "triton_url": TRITON_URL,
        "prompt_file": str(prompt_file),
        "setting_label": SETTING_LABEL,
        "max_new_tokens": max_tokens,
        "qps_levels": QPS_LEVELS,
        "test_duration_sec": TEST_DURATION_SEC,
        "grpc_mode": "streaming",
    }

    metadata_file = run_dir / "run_metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    latest_summary = RESULTS_BASE_DIR / "latest_phase5_triton_qps_summary.csv"
    summary_df.to_csv(latest_summary, index=False)

    print(f"\nSaved request log to: {request_log_file}")
    print(f"Saved QPS summary to: {summary_file}")
    print(f"Saved metadata to: {metadata_file}")
    print(f"Saved latest summary to: {latest_summary}")


if __name__ == "__main__":
    main()
