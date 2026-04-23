#!/usr/bin/env bash
set -euo pipefail

OUT_FILE="${1:-gpu_stats.csv}"

nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv \
  -l 1 > "$OUT_FILE"