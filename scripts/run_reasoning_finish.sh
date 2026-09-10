#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/.modelscope}"
PYTHON="$ROOT/.venv_kvreuse/bin/python"
RUNNER="$ROOT/scripts/run_direct_reuse.py"

run_if_needed() {
  local gpu="$1" model="$2" input="$3" root="$4" method="$5" expected="$6"
  local out="$root/$model/$method"
  local summary="$out/qwen3-$model/summary.json"
  if [[ -f "$summary" ]]; then
    local n
    n=$(python3 -c "import json; print(json.load(open('$summary'))['metrics'].get('completed_samples',0))")
    if [[ "$n" -ge "$expected" ]]; then
      echo "[GPU$gpu] skip $out ($n/$expected)"
      return 0
    fi
  fi
  echo "[GPU$gpu] run $out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$RUNNER" \
    --model "$model" \
    --input "$input" \
    --output-root "$out" \
    --method "$method" \
    --explicit-reasoning \
    --boxed-output \
    --max-new-tokens 1024 \
    --no-plots \
    --overwrite
}

run_dataset() {
  local gpu="$1" input="$2" root="$3" expected="$4"
  for model in 1.7b 4b; do
    run_if_needed "$gpu" "$model" "$input" "$root" full "$expected"
    run_if_needed "$gpu" "$model" "$input" "$root" reuse "$expected"
  done
}

run_dataset 0 data/benchmark/benchmark_job_interview_110.jsonl results/job_interview_110_reasoning 110 &
run_dataset 1 data/benchmark/benchmark_fantom_110.jsonl results/fantom_110_reasoning 110 &
run_dataset 2 data/benchmark/benchmark_perspectrum_110.jsonl results/perspectrum_110_reasoning 110 &
run_dataset 3 data/benchmark/benchmark_explore_tom_44.jsonl results/explore_tom_44_reasoning 44 &
wait
echo "All reasoning runs finished."
