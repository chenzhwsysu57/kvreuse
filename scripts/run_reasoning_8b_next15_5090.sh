#!/usr/bin/env bash
# Sequential 8B visible-reasoning queue: 8 methods on the 301 benchmark,
# then 7 methods on the 266-record HelpSteer2 / PKU-SafeRLHF benchmark.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SIZE="8b"
REASONING="yes"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"

cd "$ROOT"

run_301() {
  local method="$1"
  printf '[%s] starting 301 %s\n' "$(date '+%F %T %z')" "$method"
  "$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
    --method "$method" --reasoning "$REASONING" --model "$MODEL_SIZE" \
    --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
    --output-root results/benchmark_argkp_deal_harmbench_301 \
    "${RUNTIME_ARGS[@]}"
}

run_266() {
  local method="$1"
  printf '[%s] starting 266 %s\n' "$(date '+%F %T %z')" "$method"
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size "$MODEL_SIZE" --reasoning on --method "$method" \
    "${RUNTIME_ARGS[@]}"
}

for method in clean_reuse tail16_recompute tail16_post_recompute ours_post kvcomm relaycaching cacheblend epic; do
  run_301 "$method"
done

for method in reuse clean_reuse tail16_recompute tail16_post_recompute ours_post kvcomm relaycaching; do
  run_266 "$method"
done

printf '[%s] completed 15-task 8B reasoning queue\n' "$(date '+%F %T %z')"
