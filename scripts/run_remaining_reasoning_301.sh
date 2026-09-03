#!/usr/bin/env bash
# Continue the 301-record visible-reasoning benchmark with the three methods
# that were not produced by the original method suite.
# Example: bash scripts/run_remaining_reasoning_301.sh --model-size 1.7b
set -euo pipefail

usage() {
  echo "Usage: $0 --model-size {1.7b|4b}" >&2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SIZE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-size) MODEL_SIZE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
case "$MODEL_SIZE" in 1.7b|4b) ;; *) usage; exit 2 ;; esac

KVREUSE_PYTHON="${KVREUSE_PYTHON:-/home/czw/miniconda3/envs/kvreuse/bin/python}"
RELAY_PYTHON="${RELAY_PYTHON:-/home/czw/miniconda3/envs/relaycaching/bin/python}"
INPUT="$ROOT/data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl"
OUTPUT_ROOT="$ROOT/results/benchmark_argkp_deal_harmbench_301"

cd "$ROOT"
for method in cacheblend epic kvcomm; do
  "$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
    --method "$method" \
    --reasoning yes \
    --model "$MODEL_SIZE" \
    --input "$INPUT" \
    --output-root "$OUTPUT_ROOT" \
    --kvreuse-python "$KVREUSE_PYTHON" \
    --relay-python "$RELAY_PYTHON"
done

"$KVREUSE_PYTHON" -u analysis/summarize_benchmark_methods.py \
  --root "$OUTPUT_ROOT/reasoning" \
  --model-dir "qwen3-$MODEL_SIZE" \
  --methods full reuse clean_reuse tail16_recompute tail16_post_recompute ours_post relaycaching cacheblend epic kvcomm \
  --output "$ROOT/analysis/outputs/benchmark_argkp_deal_harmbench_301_${MODEL_SIZE}_reasoning_method_accuracy.md"
