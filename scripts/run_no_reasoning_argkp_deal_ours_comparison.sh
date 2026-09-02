#!/usr/bin/env bash
# Evaluate only the two new target-prefix-repeat variants on the 110 ArgKP +
# 110 Deal subset, then summarize them beside the existing ten-method results.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"

INPUT="$ROOT/data/benchmark/benchmark_argkp_deal_220.jsonl"
OUTPUT_ROOT="$ROOT/results/benchmark_argkp_deal_harmbench_301"
NEW_METHODS=(ours_repeat_txt ours_repeat_kv)
SUMMARY_METHODS=(
  full reuse clean_reuse tail16_recompute tail16_post_recompute ours_post
  relaycaching cacheblend epic kvcomm ours_repeat_txt ours_repeat_kv
)

cd "$ROOT"
# Keep a fixed, versioned 110+110 benchmark so every model and method sees the
# same directions.  Do not silently replace an existing artifact.
if [[ ! -f "$INPUT" ]]; then
  "$KVREUSE_PYTHON" -u scripts/prepare_benchmark.py \
    --datasets argkp deal_or_no_deal --per-dataset 110 --output "$INPUT"
fi

for model in 1.7b 4b 8b; do
  for method in "${NEW_METHODS[@]}"; do
    "$KVREUSE_PYTHON" -u scripts/run_benchmark.py \
      --method "$method" --reasoning no --model "$model" \
      --input "$INPUT" --output-root "$OUTPUT_ROOT" \
      "${RUNTIME_ARGS[@]}"
  done
  # Older comparison runs are not necessarily available for every model size.
  # Summarize only methods with a completed artifact for this model, instead of
  # failing after the newly requested runs have already completed.
  AVAILABLE_METHODS=()
  for method in "${SUMMARY_METHODS[@]}"; do
    if [[ -f "$OUTPUT_ROOT/no_reasoning/$method/qwen3-$model/samples.jsonl" ]]; then
      AVAILABLE_METHODS+=("$method")
    else
      echo "[summary] skipping missing artifact: $method / qwen3-$model" >&2
    fi
  done
  if (( ${#AVAILABLE_METHODS[@]} == 0 )); then
    echo "[summary] no completed artifacts found for qwen3-$model" >&2
    continue
  fi
  "$KVREUSE_PYTHON" -u analysis/summarize_benchmark_methods.py \
    --root "$OUTPUT_ROOT/no_reasoning" --model-dir "qwen3-$model" \
    --methods "${AVAILABLE_METHODS[@]}" \
    --datasets argkp deal_or_no_deal \
    --output "$ROOT/analysis/outputs/benchmark_argkp_deal_220_${model}_no_reasoning_12_method_accuracy.md"
done
