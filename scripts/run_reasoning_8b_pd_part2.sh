#!/usr/bin/env bash
# 8B reasoning PD queue, Lq-Sakura GPU 1: two 301 and two 266 methods.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"
cd "$ROOT"

for method in relaycaching cacheblend; do
  "$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
    --method "$method" --reasoning yes --model 8b \
    --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
    --output-root results/benchmark_argkp_deal_harmbench_301 \
    --overwrite \
    "${RUNTIME_ARGS[@]}"
done

for method in tail16_post_recompute ours_post; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 8b --reasoning on --method "$method" --overwrite "${RUNTIME_ARGS[@]}"
done
