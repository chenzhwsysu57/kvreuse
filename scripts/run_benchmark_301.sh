#!/usr/bin/env bash
# Run the requested method suite on the 301-record benchmark.
# Example: bash scripts/run_benchmark_301.sh --model-size 4b --reasoning on
set -euo pipefail

usage() {
  echo "Usage: $0 --model-size {1.7b|4b|8b} --reasoning {on|off}" >&2
}

MODEL_SIZE=""
REASONING=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-size)
      MODEL_SIZE="${2:-}"
      shift 2
      ;;
    --reasoning)
      REASONING="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$MODEL_SIZE" in 1.7b|4b|8b) ;; *) usage; exit 2 ;; esac
case "$REASONING" in on|off) ;; *) usage; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$ROOT/data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl"
OUTPUT_ROOT="$ROOT/results/benchmark_argkp_deal_harmbench_301"
PYTHON_BIN="${KVREUSE_PYTHON:-/home/czw/miniconda3/envs/kvreuse/bin/python}"
if [[ "$REASONING" == "on" ]]; then
  REASONING_ARG="yes"
  MODE_DIR="reasoning"
else
  REASONING_ARG="no"
  MODE_DIR="no_reasoning"
fi

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
    --reasoning "$REASONING_ARG" \
    --model "$MODEL_SIZE" \
    --input "$INPUT" \
    --output-root "$OUTPUT_ROOT"
done

"$PYTHON_BIN" -u analysis/summarize_benchmark_methods.py \
  --root "$OUTPUT_ROOT/$MODE_DIR" \
  --model-dir "qwen3-$MODEL_SIZE" \
  --methods "${METHODS[@]}" \
  --output "$ROOT/analysis/outputs/benchmark_argkp_deal_harmbench_301_${MODEL_SIZE}_${MODE_DIR}_method_accuracy.md"
