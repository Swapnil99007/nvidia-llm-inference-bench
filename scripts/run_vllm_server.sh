#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-distilgpt2}"
HOST="${2:-0.0.0.0}"
PORT="${3:-8000}"
API_KEY="${4:-token-abc123}"

vllm serve "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY"