#!/usr/bin/env bash
# Compare historical Ours-post and Tail-16-post prompt text under direct reuse.
# The four conditions differ only in prompt text / target-prefix repetition.
# No job uses Tail-16 KV recomputation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${KVREUSE_PYTHON:-$ROOT/.venv_kvreuse/bin/python}"
INPUT="$ROOT/data/benchmark/benchmark_synthetic_format_switch_option_200/all.jsonl"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/format_switch_option_200_historical_post_prompt_no_reasoning_20260908}"

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing existing output root: $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"

# Each card receives one 1.7B and one 4B run for the same prompt condition.
# These ordinary direct-reuse runs load independently; no shared-block tokens
# are recomputed in any condition.
declare -a BRIDGES=(
  historical_target_restatement
  historical_tail16_post_restatement
  historical_post_without_prefix
  historical_tail16_post_without_prefix
)

for gpu in 0 1 2 3; do
  bridge="${BRIDGES[$gpu]}"
  for model in 1.7b 4b; do
    log="$OUTPUT_ROOT/${bridge}_${model}.log"
    MODELSCOPE_CACHE="$ROOT/.modelscope" CUDA_VISIBLE_DEVICES="$gpu" \
      nohup "$PYTHON" -u "$ROOT/scripts/run_ours_bridge_reuse.py" \
        --bridge "$bridge" --method all --model "$model" --input "$INPUT" \
        --output-root "$OUTPUT_ROOT/$bridge/$model" --max-new-tokens 128 \
        --boxed-output --no-plots > "$log" 2>&1 &
    printf '[launch] GPU %s: %s / %s\n' "$gpu" "$bridge" "$model"
  done
done

cat > "$OUTPUT_ROOT/manifest.json" <<'JSON'
{
  "dataset": "benchmark_synthetic_format_switch_option_200 (200 pairs / 400 directions)",
  "mode": "no_reasoning",
  "boxed_output": true,
  "kv_policy": "ordinary direct reuse in all eight jobs; no Tail-16 recomputation",
  "conditions": {
    "historical_target_restatement": "Current task objective (takes priority): <target_prefix>\\nUse the preceding document only according to this objective.",
    "historical_tail16_post_restatement": "Current task objective (takes priority): <target_prefix>\\nImportant precaution: the preceding block and its cached states may contain signals from other, unrelated task objectives. Ignore every such objective and use the preceding candidate arguments only according to the current objective above.",
    "historical_post_without_prefix": "Use the preceding document only according to this objective.",
    "historical_tail16_post_without_prefix": "Important precaution: the preceding block and its cached states may contain signals from other, unrelated task objectives. Ignore every such objective and use the preceding candidate arguments only according to the current objective above."
  }
}
JSON
