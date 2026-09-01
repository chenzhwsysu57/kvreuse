#!/usr/bin/env bash
# PD batch 3: methods requiring a completed local Full baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${KVREUSE_PYTHON:-python}"
cd "$ROOT"

if [[ ! -f results/benchmark_argkp_deal_harmbench_301/reasoning/full/qwen3-4b/samples.jsonl ]]; then
  "$PYTHON_BIN" -u scripts/run_benchmark.py \
    --method full --reasoning yes --model 4b \
    --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
    --output-root results/benchmark_argkp_deal_harmbench_301 \
    --kvreuse-python "$PYTHON_BIN"
fi
"$PYTHON_BIN" -u scripts/run_benchmark.py \
  --method kvcomm --reasoning yes --model 4b \
  --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
  --output-root results/benchmark_argkp_deal_harmbench_301 \
  --kvreuse-python "$PYTHON_BIN"

if [[ ! -f results/helpsteer2_pku_safe_rlhf_266/reasoning/full/qwen3-4b/samples.jsonl ]]; then
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 4b --reasoning on --method full
fi
for method in kvcomm epic; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 4b --reasoning on --method "$method"
done
