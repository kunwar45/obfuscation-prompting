#!/usr/bin/env bash
set -euo pipefail

# Reproducible tiny end-to-end run against a local OpenAI-compatible endpoint.
# Usage:
#   bash scripts/run_local_e2e_smoke.sh
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct DATASET=concealment bash scripts/run_local_e2e_smoke.sh

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MONITOR_MODEL="${MONITOR_MODEL:-$MODEL}"
DATASET="${DATASET:-concealment}"   # concealment | json
PROVIDER="${PROVIDER:-local-openai}" # local-openai | mock
API_BASE="${LOCAL_API_BASE:-http://127.0.0.1:8000/v1}"
OUT_DIR="${OUT_DIR:-results/smoke_local}"

mkdir -p "$OUT_DIR"

if [[ "$DATASET" == "concealment" ]]; then
  python3 main.py \
    --provider "$PROVIDER" \
    --local-api-base "$API_BASE" \
    --local-api-key local \
    --dataset concealment \
    --concealment-file data/smoke_concealment.jsonl \
    --concealment-conditions A0,A1 \
    --concealment-query-types B1 \
    --limit 3 \
    --base-model "$MODEL" \
    --monitor-model "$MONITOR_MODEL" \
    --disable-llm-monitor \
    --max-tokens 256 \
    --temperature 0.0 \
    --output-dir "$OUT_DIR"
elif [[ "$DATASET" == "json" ]]; then
  python3 main.py \
    --provider "$PROVIDER" \
    --local-api-base "$API_BASE" \
    --local-api-key local \
    --prompts data/smoke_prompts.json \
    --base-model "$MODEL" \
    --monitor-model "$MONITOR_MODEL" \
    --disable-llm-monitor \
    --max-tokens 256 \
    --temperature 0.0 \
    --output-dir "$OUT_DIR"
else
  echo "Unsupported DATASET=$DATASET (expected: concealment or json)" >&2
  exit 1
fi

echo "Smoke run completed. Results in: $OUT_DIR"
