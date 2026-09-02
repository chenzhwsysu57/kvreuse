#!/usr/bin/env bash
# PD batch 3: methods requiring a completed local Full baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"
cd "$ROOT"

# Build a local 301 Full artifact when this batch is dispatched to a machine
# that does not already have one, then run its dependent KVCOMM method.
if [[ ! -f results/benchmark_argkp_deal_harmbench_301/reasoning/full/qwen3-1.7b/samples.jsonl ]]; then
  "$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
    --method full --reasoning yes --model 1.7b \
    --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
    --output-root results/benchmark_argkp_deal_harmbench_301 \
    --overwrite \
    "${RUNTIME_ARGS[@]}"
fi
"$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
  --method kvcomm --reasoning yes --model 1.7b \
  --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
  --output-root results/benchmark_argkp_deal_harmbench_301 \
  --overwrite \
  "${RUNTIME_ARGS[@]}"

# The 266 reasoning Full result is currently recorded from Lq-Sakura only.
# Build a local artifact only when needed, then run its dependent KVCOMM.
if [[ ! -f results/helpsteer2_pku_safe_rlhf_266/reasoning/full/qwen3-1.7b/samples.jsonl ]]; then
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 1.7b --reasoning on --method full --overwrite "${RUNTIME_ARGS[@]}"
fi

for method in kvcomm epic; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 1.7b --reasoning on --method "$method" --overwrite "${RUNTIME_ARGS[@]}"
done
