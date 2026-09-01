#!/usr/bin/env bash
# PD batch 1: independent 301 EPIC plus 266 direct-reuse methods.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"
cd "$ROOT"

"$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
  --method epic --reasoning yes --model 4b \
  --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
  --output-root results/benchmark_argkp_deal_harmbench_301 \
  "${RUNTIME_ARGS[@]}"

for method in reuse clean_reuse tail16_recompute; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 4b --reasoning on --method "$method" "${RUNTIME_ARGS[@]}"
done
