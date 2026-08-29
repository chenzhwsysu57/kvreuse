#!/usr/bin/env bash
set -euo pipefail

# Run origin/reuse x no-reasoning/reasoning on the same validation records.
# Usage: bash scripts/run_reasoning_ablation.sh [model_size] [input_jsonl]

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

model_size="${1:-1.7b}"
input_file="${2:-data/validation/second_step_10_each_harmbench.jsonl}"
python_bin="${KVREUSE_PYTHON:-/home/czw/miniconda3/envs/kvreuse/bin/python}"
output_root="${KVREUSE_ABLATION_ROOT:-results/reasoning_ablation}"

if [[ ! -x "$python_bin" ]]; then
  python_bin="python"
fi
if [[ ! -f "$input_file" ]]; then
  echo "input file not found: $input_file" >&2
  exit 1
fi

common=(
  --model "$model_size"
  --input "$input_file"
  --max-samples 0
  --no-plots
  --overwrite
)

echo "[$model_size] origin/reuse x no-reasoning"
"$python_bin" -u scripts/run_direct_reuse.py \
  "${common[@]}" \
  --output-root "$output_root/no_reasoning" \
  --max-new-tokens 32 \
  --boxed-output

echo "[$model_size] origin/reuse x explicit reasoning"
"$python_bin" -u scripts/run_direct_reuse.py \
  "${common[@]}" \
  --output-root "$output_root/reasoning" \
  --max-new-tokens 1024 \
  --explicit-reasoning \
  --boxed-output

echo "ablation complete"
echo "no-reasoning: $output_root/no_reasoning/qwen3-$model_size/summary.json"
echo "reasoning:    $output_root/reasoning/qwen3-$model_size/summary.json"
