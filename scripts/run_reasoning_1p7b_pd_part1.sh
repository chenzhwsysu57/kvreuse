#!/usr/bin/env bash
# PD batch 1: independent 301 EPIC plus 266 direct-reuse methods.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KVREUSE_PYTHON:-python}"
cd "$ROOT"

"$PYTHON_BIN" -u scripts/run_benchmark.py \
  --method epic --reasoning yes --model 1.7b \
  --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
  --output-root results/benchmark_argkp_deal_harmbench_301 \
  --kvreuse-python "$PYTHON_BIN"

for method in reuse clean_reuse tail16_recompute; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 1.7b --reasoning on --method "$method"
done
