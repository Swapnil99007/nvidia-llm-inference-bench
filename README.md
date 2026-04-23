# LLM Inference Optimization and Benchmarking on NVIDIA GPUs

This project benchmarks and compares different large language model inference paths, starting from a lightweight local baseline and evolving into a reproducible benchmarking framework for modern LLM serving systems.

The goal of the project is to study how different inference stacks behave under controlled workloads, with a focus on:
- latency
- throughput
- output length consistency
- serving-system behavior
- reproducibility of benchmarking workflows

The project currently compares:
- a direct Hugging Face Transformers baseline
- a vLLM server-based inference workflow

and is designed to be extended later to:
- concurrency/load testing
- TensorRT-LLM
- Triton Inference Server
- larger-scale NVIDIA GPU benchmarking

---

## Repository Structure

```text
nvidia-llm-inference-bench/
├── README.md
├── requirements.txt
├── configs/
│   ├── model_config.yaml
│   └── benchmark_matrix.yaml
├── prompts/
│   └── prompts.jsonl
├── scripts/
│   ├── run_baseline.py
│   ├── run_benchmark.py
│   ├── summarize_results.py
│   ├── plot_phase2_results.py
│   ├── run_vllm_server.sh
│   ├── run_vllm_benchmark.py
│   ├── compare_engines.py
│   └── plot_engine_comparison.py
├── results/
│   ├── raw/
│   └── figures/
└── report/

## Phase 1: Local Baseline

Phase 1 establishes the initial benchmarking workflow on a local development machine.

The purpose of this phase was not to achieve strong model quality, but to build the core benchmarking pipeline:
- prompt loading
- model execution
- latency measurement
- throughput calculation
- CSV logging
- plot generation

### Baseline setup
- Model: `distilgpt2`
- Device: Apple Silicon MPS or CPU
- Prompt set: 20 prompts across:
  - short Q&A
  - summarization
  - coding
  - reasoning
  - long-context tasks

### Metrics logged
- input token count
- output token count
- latency
- tokens per second

### Output artifacts
- `results/raw/baseline_results.csv`
- `results/figures/latency_by_prompt.png`
- `results/figures/tokens_per_sec_by_prompt.png`
- `results/figures/avg_latency_by_category.png`

### Phase 1 observations
- Built a local benchmark harness for causal language model inference
- Logged per-prompt latency and throughput to structured CSV output
- Generated baseline plots for latency and throughput analysis
- Established a reproducible workflow that could later be extended to vLLM and other inference engines

### Why this phase matters
Phase 1 validated the benchmarking pipeline itself.

Although `distilgpt2` is not a strong instruction-following model, it was lightweight enough to verify that the project structure, logging, and visualization flow were working correctly.

## Phase 2: Config-Driven Benchmark Framework

Phase 2 refactors the initial local benchmark into a reusable, config-driven framework.

Instead of a one-off script, the benchmark pipeline was upgraded to support:
- YAML-based configuration
- timestamped run directories
- structured metadata logging
- reproducible result summaries
- per-run figure generation

### Improvements over Phase 1
- Added YAML-based model and benchmark configuration
- Added timestamped run directories for reproducibility
- Logged run metadata to JSON
- Added aggregate summary generation by setting and category
- Extended plotting for structured per-run analysis

### Benchmark dimensions
- Model: `distilgpt2`
- Settings:
  - short output (`max_new_tokens=32`)
  - default output (`max_new_tokens=64`)
  - long output (`max_new_tokens=96`)

### Output artifacts
Each run produces:
- `benchmark_results.csv`
- `run_metadata.json`
- `run_summary.json`
- `summary_by_setting_and_category.csv`
- per-run plots under `results/figures/<run_dir>/`

### Purpose
This phase turned the project from a simple local experiment into a reusable benchmarking framework that could be extended to compare multiple inference engines under consistent settings.


---

## Phase 3: Engine Comparison with Hugging Face Transformers and vLLM

Phase 3 upgrades the project from a single-engine benchmark into an actual inference-system comparison.

In this phase, the project compares:
- `hf_transformers`: direct in-process generation using Hugging Face Transformers
- `vllm`: server-based inference using the vLLM OpenAI-compatible API

### Hardware and environment
- GPU: NVIDIA GeForce RTX 3090
- Environment: Linux cloud GPU instance
- Model: `Qwen/Qwen2.5-7B-Instruct`

### Why this phase is important
This is the first phase where the project starts to resemble a real ML systems benchmarking workflow rather than a local prototype.

It answers a more meaningful question:

> How does a direct Transformers baseline compare with a serving-oriented engine like vLLM when both run the same instruct model under the same prompt set and output budgets?

### Benchmark settings
The following output budgets are currently evaluated:
- short output (`max_new_tokens=32`)
- default output (`max_new_tokens=64`)
- long output (`max_new_tokens=96`)

### Prompt categories
The same prompt set is used across both engines:
- short Q&A
- summarization
- coding
- reasoning
- long-context prompts

### Metrics compared
- average latency
- tokens per second
- average input token count
- average output token count
- number of prompts per category

### Key implementation details
- Hugging Face baseline uses direct model inference
- vLLM uses a server-based inference path
- token counting was aligned across both engines using the same tokenizer
- finish reasons were inspected on the vLLM side to verify whether generation was ending due to:
  - `length`
  - natural `stop`

### Phase 3 results
After aligning token counting across engines, the comparison became much more reliable.

Key observations:
- vLLM consistently achieved lower latency than the Hugging Face baseline across the tested settings
- vLLM also achieved higher throughput (tokens/sec) across most categories
- output token counts became closely aligned between both engines after the tokenizer fix
- most vLLM generations ended due to `length`, meaning the model was typically reaching the requested output cap rather than stopping prematurely

### Example findings
Across the Qwen2.5-7B-Instruct runs on RTX 3090:
- for short output settings, vLLM generally reduced latency relative to the HF baseline
- for default and long output settings, vLLM maintained higher throughput while producing comparable output lengths
- summarization prompts sometimes stopped slightly earlier than the full output budget, which is expected behavior for an instruct-tuned model

### Phase 3 output artifacts
- per-engine run folders under `results/raw/`
- `results/raw/latest_engine_comparison_summary.csv`
- engine comparison plots under `results/figures/engine_comparison/`

### What Phase 3 demonstrates
Phase 3 shows that this project can now:
- benchmark a modern instruct model
- compare two inference engines under controlled settings
- surface measurable engine-level differences
- generate artifacts suitable for project documentation and future resume bullets


### Phase 3.1: vLLM Concurrency Benchmark

To extend the single-request engine comparison, a concurrency benchmark was added for the vLLM serving path using `Qwen/Qwen2.5-7B-Instruct` on an RTX 3090.

Concurrency levels tested:
- 1
- 2
- 4
- 8

Requests per level:
- 16

Key observations:
- average latency remained almost flat from concurrency levels 1 to 4
- latency increased only slightly at concurrency 8
- average output length remained stable across all concurrency levels
- the results suggest that vLLM handled moderate parallel request load efficiently without a major latency blow-up

Output artifacts:
- `results/raw/phase31_vllm_concurrency_qwen25_7b_instruct_<timestamp>/benchmark_results.csv`
- `results/raw/phase31_vllm_concurrency_qwen25_7b_instruct_<timestamp>/concurrency_summary.csv`
- `results/figures/<run_dir>/latency_vs_concurrency.png`
- `results/figures/<run_dir>/tokens_per_sec_vs_concurrency.png`

---

## Phase 4: TensorRT-LLM Integration and Full Engine Comparison

Phase 4 extends the benchmarking framework to include NVIDIA TensorRT-LLM, enabling a full comparison across three inference engines:

- `hf_transformers`
- `vllm`
- `tensorrt_llm`

### Hardware and Setup
- GPU: NVIDIA RTX 3090
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Same prompt set and benchmark configuration as Phase 3

### Benchmark Scope
- Output lengths:
  - short (32 tokens)
  - default (64 tokens)
  - long (96 tokens)
- Prompt categories:
  - coding
  - reasoning
  - summarization
  - short QA
  - long-context

### Key Results

#### Throughput (tokens/sec)

| Engine        | Short | Default | Long |
|--------------|------|--------|------|
| HF           | ~42  | ~42–43 | ~40–43 |
| vLLM         | ~48–50 | ~50.3 | ~50.4 |
| TensorRT-LLM | ~50  | ~50.7 | ~50.7 |

#### Latency

| Engine        | Short | Default | Long |
|--------------|------|--------|------|
| HF           | ~0.74s | ~1.50s | ~2.2–2.4s |
| vLLM         | ~0.64–0.80s | ~1.27s | ~1.90s |
| TensorRT-LLM | ~0.63s | ~1.26s | ~1.89s |

### Observations

- TensorRT-LLM achieves the highest and most consistent throughput across all workloads
- vLLM closely matches TensorRT performance but shows instability in short-output scenarios
- Hugging Face baseline is consistently slower and scales poorly with output length
- TensorRT-LLM demonstrates near-flat throughput across categories, indicating strong GPU kernel optimization

### Insights

- TensorRT-LLM benefits from kernel fusion and optimized GPU execution
- vLLM leverages KV cache and batching but introduces serving overhead
- Hugging Face lacks serving optimizations, resulting in lower efficiency

### Output Artifacts

- `results/raw/latest_engine_comparison_summary.csv`
- `results/figures/engine_comparison/*.png`

### What Phase 4 Demonstrates

Phase 4 elevates the project into a full ML systems benchmarking study by:

- comparing three inference stacks under identical conditions
- analyzing system-level performance differences
- producing reproducible, quantitative results

## Current Status

The project currently supports:
- local baseline benchmarking
- config-driven benchmark execution
- structured metadata logging
- Hugging Face vs vLLM vs TensorRT-LLM comparison on a modern instruct model
- per-run summaries and comparison plots

---

## Limitations

TensorRT-LLM is integrated for single-request benchmarking, but:
- concurrency benchmarking for TensorRT-LLM is not yet implemented
- Triton Inference Server integration is pending

---

## Next Planned Improvements

With Phase 4 completing a full multi-engine comparison (HF vs vLLM vs TensorRT-LLM), the next steps focus on scaling this project toward production-grade inference benchmarking.

Planned upgrades include:

- TensorRT-LLM concurrency benchmarking
  - evaluate multi-request performance under parallel load
  - compare scaling behavior against vLLM

- request-rate / load testing
  - simulate real-world traffic patterns (QPS-based benchmarking)
  - measure latency degradation under sustained load

- Triton Inference Server integration
  - deploy TensorRT-LLM via Triton
  - benchmark production-style serving pipelines

- end-to-end serving system comparison
  - vLLM vs Triton vs custom pipelines
  - include batching, scheduling, and queueing behavior

- GPU utilization and memory profiling
  - analyze VRAM usage across engines
  - correlate memory efficiency with throughput

- expanded benchmark dimensions
  - longer context windows (8k, 16k, 32k)
  - larger models (13B+)
  - mixed workload scenarios

- stability and tail-latency analysis
  - measure P95/P99 latency
  - identify jitter and variance across engines

---

## Summary

This project started as a lightweight local benchmarking workflow and has evolved into a reproducible inference benchmarking framework for modern LLM serving systems.

Across four phases, the project progressed from:
- a local baseline (Phase 1)
- to a config-driven benchmarking framework (Phase 2)
- to serving-system comparison with vLLM (Phase 3)
- to a full multi-engine GPU benchmark including TensorRT-LLM (Phase 4)

The most important result is the Phase 4 comparison:

- same model
- same prompts
- same hardware
- three different inference engines
- clear, measurable differences in latency, throughput, and stability

Key findings:
- TensorRT-LLM achieves the highest and most consistent throughput (~50 tok/s)
- vLLM closely matches TensorRT performance but introduces occasional instability in small workloads
- Hugging Face Transformers baseline is consistently slower (~15–20%) and scales poorly with output length

This project now reflects real-world ML systems engineering concerns:
- GPU efficiency
- inference optimization
- serving architecture trade-offs
- reproducible benchmarking workflows

It provides a strong foundation for further NVIDIA-aligned work in high-performance inference systems, including Triton deployment and large-scale serving optimization.