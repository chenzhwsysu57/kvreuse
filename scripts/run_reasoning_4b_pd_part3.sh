#!/usr/bin/env bash
# PD batch 3: methods requiring a completed local Full baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"
cd "$ROOT"

if [[ ! -f results/benchmark_argkp_deal_harmbench_301/reasoning/full/qwen3-4b/samples.jsonl ]]; then
  "$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
    --method full --reasoning yes --model 4b \
    --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
    --output-root results/benchmark_argkp_deal_harmbench_301 \
    "${RUNTIME_ARGS[@]}"
fi
"$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
  --method kvcomm --reasoning yes --model 4b \
  --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
  --output-root results/benchmark_argkp_deal_harmbench_301 \
  "${RUNTIME_ARGS[@]}"

if [[ ! -f results/helpsteer2_pku_safe_rlhf_266/reasoning/full/qwen3-4b/samples.jsonl ]]; then
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 4b --reasoning on --method full "${RUNTIME_ARGS[@]}"
fi
for method in kvcomm epic; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 4b --reasoning on --method "$method" "${RUNTIME_ARGS[@]}"
done
