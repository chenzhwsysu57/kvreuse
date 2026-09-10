#!/usr/bin/env bash
# Tests generic post length independently of target-specific task information.
# All post prompts are shared across every example and name neither JSON/CSV,
# target row, operation, option, nor target answer.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${KVREUSE_PYTHON:-$ROOT/.venv_kvreuse/bin/python}"
INPUT="$ROOT/data/benchmark/benchmark_synthetic_format_switch_option_200/all.jsonl"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/format_switch_option_200_generic_post_length_no_reasoning_20260908}"
PROMPTS="$ROOT/scripts/generic_post_prompts"

[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing existing output root: $OUTPUT_ROOT" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"

for index in 0 1 2 3; do
  name=(short guided_48 guided_96 guided_160)
  prompt="${name[$index]}"
  for model in 1.7b 4b; do
    MODELSCOPE_CACHE="$ROOT/.modelscope" CUDA_VISIBLE_DEVICES="$index" \
      nohup "$PYTHON" -u "$ROOT/kv_semantic_attack/eval_synthetic_batch.py" \
        --input "$INPUT" --output "$OUTPUT_ROOT/$prompt/$model/evaluation.json" \
        --model "$model" --method reuse --reuse-engine scatter --batch-size 16 \
        --max-new-tokens 128 --suffix-file "$PROMPTS/$prompt.txt" \
        > "$OUTPUT_ROOT/${prompt}_${model}.log" 2>&1 &
    printf '[launch] GPU %s: generic-%s / %s\n' "$index" "$prompt" "$model"
  done
done

"$PYTHON" - <<PY
import json
from pathlib import Path
root = Path("$OUTPUT_ROOT")
entries = {}
for path in sorted(Path("$PROMPTS").glob("*.txt")):
    entries[path.stem] = path.read_text(encoding="utf-8").strip()
(root / "manifest.json").write_text(json.dumps({
    "dataset": str(Path("$INPUT")), "mode": "no_reasoning", "boxed_output": True,
    "method": "direct reuse + target-independent post-block suffix",
    "models": ["1.7b", "4b"], "prompts": entries,
    "control": "All suffixes omit task-specific operation, format, row, options, and answer."
}, indent=2) + "\n", encoding="utf-8")
PY
