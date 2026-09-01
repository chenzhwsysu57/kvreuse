#!/usr/bin/env bash
# Run the requested 4B visible-reasoning methods on the 301-record benchmark.
# Safe to rerun: every runner resumes completed records by default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$ROOT/data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl"
OUTPUT_ROOT="$ROOT/results/benchmark_argkp_deal_harmbench_301"
PYTHON_BIN="${KVREUSE_PYTHON:-/home/czw/miniconda3/envs/kvreuse/bin/python}"

METHODS=(
  full
  reuse
  clean_reuse
  tail16_recompute
  tail16_post_recompute
  ours_post
  relaycaching
  cacheblend
  epic
  kvcomm
)

cd "$ROOT"
for method in "${METHODS[@]}"; do
  "$PYTHON_BIN" -u scripts/run_benchmark.py \
    --method "$method" \
    --reasoning yes \
    --model 4b \
    --input "$INPUT" \
    --output-root "$OUTPUT_ROOT"
done

"$PYTHON_BIN" -u analysis/summarize_benchmark_methods.py \
  --root "$OUTPUT_ROOT/reasoning" \
  --model-dir qwen3-4b \
  --methods "${METHODS[@]}" \
  --output "$ROOT/analysis/outputs/benchmark_argkp_deal_harmbench_301_4b_reasoning_method_accuracy.md"
