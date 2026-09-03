#!/usr/bin/env bash
# Run the currently unstarted 8B visible-reasoning tasks for the 266 benchmark.
set -euo pipefail

ROOT="${KVREUSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KVREUSE_PYTHON="${KVREUSE_PYTHON:?KVREUSE_PYTHON must be set}"
RELAY_PYTHON="${RELAY_PYTHON:?RELAY_PYTHON must be set}"

cd "$ROOT"
for method in cacheblend epic; do
  printf '[%s] starting %s\n' "$(date '+%F %T %z')" "$method"
  "$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
    --method "$method" --reasoning yes --model 8b \
    --input data/benchmark/benchmark_helpsteer2_pku_safe_rlhf_266.jsonl \
    --output-root results/helpsteer2_pku_safe_rlhf_266 \
    --kvreuse-python "$KVREUSE_PYTHON" --relay-python "$RELAY_PYTHON"
done

printf '[%s] completed 8B reasoning PD queue\n' "$(date '+%F %T %z')"
