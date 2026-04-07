import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
BENCHMARK_CONFIG_PATH = Path("configs/benchmark_matrix.yaml")
RESULTS_BASE_DIR = Path("results/raw")


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: Path):
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def get_device(device_preference: str):
    if device_preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_preference)


def main():
    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    bench_cfg = load_yaml(BENCHMARK_CONFIG_PATH)

    run_name = bench_cfg["run_name"]
    prompt_file = Path(bench_cfg["prompt_file"])
    settings = bench_cfg["settings"]

    model_name = model_cfg["model_name"]
    warmup_runs = model_cfg["warmup_runs"]
    device = get_device(model_cfg["device_preference"])
    do_sample = model_cfg["do_sample"]

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_BASE_DIR / f"{run_name}_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    prompts = load_prompts(prompt_file)

    warmup_prompt = "Explain what machine learning inference is."
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)

    print("Starting warmup...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(
                **warmup_inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
    print("Warmup complete.")

    all_results = []

    for setting in settings:
        label = setting["label"]
        max_new_tokens = setting["max_new_tokens"]

        print(f"\nRunning setting: {label}")

        for item in prompts:
            prompt_id = item["id"]
            category = item["category"]
            prompt_text = item["prompt"]

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            input_token_count = inputs["input_ids"].shape[1]

            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )

            end_time = time.perf_counter()
            latency_sec = end_time - start_time

            generated_ids = outputs[0]
            new_token_ids = generated_ids[input_token_count:]
            output_token_count = len(new_token_ids)

            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_only_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

            tokens_per_sec = output_token_count / latency_sec if latency_sec > 0 else None

            row = {
                "run_name": run_name,
                "run_timestamp": run_timestamp,
                "engine_name": "hf_transformers",
                "setting_label": label,
                "model_name": model_name,
                "device": str(device),
                "prompt_id": prompt_id,
                "category": category,
                "input_token_count": input_token_count,
                "output_token_count": output_token_count,
                "max_new_tokens": max_new_tokens,
                "latency_sec": latency_sec,
                "tokens_per_sec": tokens_per_sec,
                "prompt_text": prompt_text,
                "output_text": output_text,
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
        "model_name": model_name,
        "device": str(device),
        "prompt_file": str(prompt_file),
        "warmup_runs": warmup_runs,
        "do_sample": do_sample,
        "num_prompts": len(prompts),
        "num_settings": len(settings),
    }
    metadata_file = run_dir / "run_metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    summary = {
        "run_name": run_name,
        "run_timestamp": run_timestamp,
        "engine_name": "hf_transformers",
        "model_name": model_name,
        "device": str(device),
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