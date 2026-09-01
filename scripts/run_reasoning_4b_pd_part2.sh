#!/usr/bin/env bash
# PD batch 2: independent RelayCaching and 266 bridge methods.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"
cd "$ROOT"

"$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
  --method relaycaching --reasoning yes --model 4b \
  --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
  --output-root results/benchmark_argkp_deal_harmbench_301 \
  "${RUNTIME_ARGS[@]}"

for method in tail16_post_recompute ours_post relaycaching cacheblend; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 4b --reasoning on --method "$method" "${RUNTIME_ARGS[@]}"
done
