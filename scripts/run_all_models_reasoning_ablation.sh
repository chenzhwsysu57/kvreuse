#!/usr/bin/env bash
set -euo pipefail

# Full current constructed dataset: all records in data/processed/all.jsonl.
# Runs both no-reasoning and explicit-reasoning for every Qwen3 size.

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

input_file="${1:-data/processed/all.jsonl}"
models=(0.6b 1.7b 4b 8b)

if [[ ! -f "$input_file" ]]; then
  echo "input file not found: $input_file" >&2
  exit 1
fi

for model in "${models[@]}"; do
  echo "========== Qwen3-$model / full dataset =========="
  bash scripts/run_reasoning_ablation.sh "$model" "$input_file"
done

echo "all-model full-dataset reasoning ablation complete"
