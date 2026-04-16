import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "distilgpt2"
PROMPTS_FILE = Path("prompts/prompts.jsonl")
OUTPUT_FILE = Path("results/raw/baseline_results.csv")
MAX_NEW_TOKENS = 64
WARMUP_RUNS = 2


def load_prompts(path: Path):
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model.to(device)
    model.eval()

    prompts = load_prompts(PROMPTS_FILE)
    results = []

    print("Starting warmup runs...")
    warmup_prompt = "Explain what machine learning inference is."
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model.generate(
                **warmup_inputs,
                max_new_tokens=16,
                do_sample=False
            )

    print("Warmup complete.")
    print(f"Running benchmark for {len(prompts)} prompts...")

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
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        end_time = time.perf_counter()
        latency_sec = end_time - start_time

        generated_ids = outputs[0]
        new_token_ids = generated_ids[input_token_count:]
        output_token_count = len(new_token_ids)

        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_only_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

        tokens_per_sec = (
            output_token_count / latency_sec if latency_sec > 0 else None
        )

        results.append({
            "prompt_id": prompt_id,
            "category": category,
            "model_name": MODEL_NAME,
            "device": str(device),
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency_sec": latency_sec,
            "tokens_per_sec": tokens_per_sec,
            "prompt_text": prompt_text,
            "generated_text": output_text,
            "generated_only_text": generated_only_text
        })

        print(
            f"[{prompt_id}] category={category} "
            f"input_tokens={input_token_count} "
            f"output_tokens={output_token_count} "
            f"latency={latency_sec:.4f}s "
            f"tok/s={tokens_per_sec:.2f}"
        )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved results to: {OUTPUT_FILE}")
    print("\nSummary:")
    print(df[["latency_sec", "tokens_per_sec", "input_token_count", "output_token_count"]].describe())


if __name__ == "__main__":
    main()