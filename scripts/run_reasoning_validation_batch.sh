#!/usr/bin/env bash
# Run 1.7b/4b reasoning full+reuse benchmarks on promising datasets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/.modelscope}"
PYTHON="${PYTHON:-$ROOT/.venv_kvreuse/bin/python}"
RUNNER="$ROOT/scripts/run_direct_reuse.py"

MODELS=(1.7b 4b)
DATASETS=(
  "job_interview:data/benchmark/benchmark_job_interview_110.jsonl:job_interview_110_reasoning"
  "fantom:data/benchmark/benchmark_fantom_110.jsonl:fantom_110_reasoning"
  "perspectrum:data/benchmark/benchmark_perspectrum_110.jsonl:perspectrum_110_reasoning"
  "explore_tom:data/benchmark/benchmark_explore_tom_44.jsonl:explore_tom_44_reasoning"
)

run_one() {
  local gpu="$1"
  local model="$2"
  local input="$3"
  local output_root="$4"
  local method="$5"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --model "$model" \
    --input "$input" \
    --output-root "$output_root" \
    --method "$method" \
    --explicit-reasoning \
    --boxed-output \
    --max-new-tokens 1024 \
    --no-plots \
    --overwrite
}

for entry in "${DATASETS[@]}"; do
  IFS=: read -r name input root <<<"$entry"
  for model in "${MODELS[@]}"; do
    echo "=== ${name} ${model} full ==="
    run_one 0 "$model" "$input" "results/${root}/${model}/full" full
    echo "=== ${name} ${model} reuse ==="
    run_one 0 "$model" "$input" "results/${root}/${model}/reuse" reuse
  done
done

echo "All reasoning validation runs completed."
