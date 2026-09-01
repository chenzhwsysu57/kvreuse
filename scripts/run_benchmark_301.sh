#!/usr/bin/env bash
# Run the requested method suite on the 301-record benchmark.
# Example: bash scripts/run_benchmark_301.sh --model-size 4b --reasoning on --method full,reuse
set -euo pipefail

usage() {
  echo "Usage: $0 --model-size {1.7b|4b|8b} --reasoning {on|off} [--method METHOD[,METHOD...]] [--input PATH] [--output-root PATH] [--relay-python PATH]" >&2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SIZE=""
REASONING=""
INPUT=""
OUTPUT_ROOT=""
RELAY_PYTHON="${RELAY_PYTHON:-}"
METHOD_REQUESTS=()
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
    --method)
      METHOD_REQUESTS+=("${2:-}")
      shift 2
      ;;
    --input)
      INPUT="${2:-}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --relay-python)
      RELAY_PYTHON="${2:-}"
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

INPUT="${INPUT:-$ROOT/data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/benchmark_argkp_deal_harmbench_301}"
PYTHON_BIN="${KVREUSE_PYTHON:-python}"
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

# Without --method preserve the historical complete suite.  A comma-separated
# list and repeated --method flags are both accepted, e.g. --method full,reuse
# or --method full --method reuse.  Keep the requested order while removing
# duplicates so that a selected method runs exactly once.
SELECTED_METHODS=()
if [[ ${#METHOD_REQUESTS[@]} -eq 0 ]]; then
  SELECTED_METHODS=("${METHODS[@]}")
  SUMMARY_NAME="benchmark_argkp_deal_harmbench_301_${MODEL_SIZE}_${MODE_DIR}_method_accuracy.md"
else
  for request in "${METHOD_REQUESTS[@]}"; do
    IFS=',' read -r -a requested <<< "$request"
    for method in "${requested[@]}"; do
      [[ -n "$method" ]] || { echo "--method cannot contain an empty name" >&2; exit 2; }
      if [[ "$method" == "all" ]]; then
        if [[ ${#METHOD_REQUESTS[@]} -ne 1 || ${#requested[@]} -ne 1 ]]; then
          echo "--method all cannot be combined with other methods" >&2
          exit 2
        fi
        SELECTED_METHODS=("${METHODS[@]}")
        break 2
      fi
      valid=0
      for known_method in "${METHODS[@]}"; do
        [[ "$method" == "$known_method" ]] && valid=1
      done
      [[ "$valid" -eq 1 ]] || { echo "Unknown method: $method" >&2; usage; exit 2; }
      already_selected=0
      for selected_method in "${SELECTED_METHODS[@]}"; do
        [[ "$method" == "$selected_method" ]] && already_selected=1
      done
      [[ "$already_selected" -eq 1 ]] || SELECTED_METHODS+=("$method")
    done
  done
  method_tag="$(IFS=-; echo "${SELECTED_METHODS[*]}")"
  SUMMARY_NAME="benchmark_argkp_deal_harmbench_301_${MODEL_SIZE}_${MODE_DIR}_${method_tag}_method_accuracy.md"
fi

cd "$ROOT"
BENCHMARK_EXTRA=(--kvreuse-python "$PYTHON_BIN")
if [[ -n "$RELAY_PYTHON" ]]; then
  BENCHMARK_EXTRA+=(--relay-python "$RELAY_PYTHON")
fi
for method in "${SELECTED_METHODS[@]}"; do
  "$PYTHON_BIN" -u scripts/run_benchmark.py \
    --method "$method" \
    --reasoning "$REASONING_ARG" \
    --model "$MODEL_SIZE" \
    --input "$INPUT" \
    --output-root "$OUTPUT_ROOT" \
    "${BENCHMARK_EXTRA[@]}"
done

"$PYTHON_BIN" -u analysis/summarize_benchmark_methods.py \
  --root "$OUTPUT_ROOT/$MODE_DIR" \
  --model-dir "qwen3-$MODEL_SIZE" \
  --methods "${SELECTED_METHODS[@]}" \
  --output "$ROOT/analysis/outputs/$SUMMARY_NAME"
