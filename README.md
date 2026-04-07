# LLM Inference Optimization and Benchmarking on NVIDIA GPUs

## Phase 1: Local Baseline

This project begins with a local baseline benchmark for text generation using a lightweight causal language model. The goal of Phase 1 is to build the benchmark pipeline, prompt dataset, result logging, and plotting workflow before moving to larger models and NVIDIA GPU-based serving systems.

### Baseline setup
- Model: `distilgpt2`
- Device: Apple Silicon MPS or CPU
- Prompt set: 20 prompts across short Q&A, summarization, coding, reasoning, and long-context tasks
- Metrics logged:
  - input token count
  - output token count
  - latency
  - tokens per second

### Output artifacts
- Raw results CSV: `results/raw/baseline_results.csv`
- Plots:
  - `results/figures/latency_by_prompt.png`
  - `results/figures/tokens_per_sec_by_prompt.png`
  - `results/figures/avg_latency_by_category.png`

### Goal of this phase
Establish a reproducible benchmark harness that can later be extended to compare multiple inference engines such as vLLM and TensorRT-LLM on NVIDIA GPUs.


### Phase 1 observations
- Built a local benchmark harness for causal language model inference
- Logged per-prompt latency and tokens/sec to structured CSV output
- Generated baseline plots for latency and throughput analysis
- Established a reusable workflow that can be extended to vLLM and TensorRT-LLM in later phases


## Phase 2: Config-Driven Benchmark Framework

Phase 2 refactors the initial local benchmark into a reusable, config-driven pipeline.

### Improvements over Phase 1
- Added YAML-based model and benchmark configuration
- Added timestamped run directories for reproducibility
- Logged run metadata to JSON
- Added aggregate summary generation by setting and category
- Extended plotting pipeline for structured per-run analysis

### Current benchmark dimensions
- Model: `distilgpt2`
- Settings:
  - short output (`max_new_tokens=32`)
  - default output (`max_new_tokens=64`)
  - long output (`max_new_tokens=96`)

### Output artifacts
Each run now creates:
- `benchmark_results.csv`
- `run_metadata.json`
- `summary_by_setting_and_category.csv`
- per-run plots under `results/figures/<run_dir>/`

### Purpose
This framework establishes the reusable benchmark harness that will later be extended to compare alternative inference engines such as vLLM and TensorRT-LLM.