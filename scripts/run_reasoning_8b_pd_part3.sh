#!/usr/bin/env bash
# 8B reasoning PD queue, Lq-Sakura GPU 2: 301 EPIC plus 266 KVCOMM/Relay.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"
cd "$ROOT"

"$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
  --method epic --reasoning yes --model 8b \
  --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
  --output-root results/benchmark_argkp_deal_harmbench_301 \
  --overwrite \
  "${RUNTIME_ARGS[@]}"

# This Full artifact is a prerequisite only; it is already recorded in CSV.
if [[ ! -f results/helpsteer2_pku_safe_rlhf_266/reasoning/full/qwen3-8b/samples.jsonl ]]; then
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 8b --reasoning on --method full --overwrite "${RUNTIME_ARGS[@]}"
fi

for method in kvcomm relaycaching; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 8b --reasoning on --method "$method" --overwrite "${RUNTIME_ARGS[@]}"
done
