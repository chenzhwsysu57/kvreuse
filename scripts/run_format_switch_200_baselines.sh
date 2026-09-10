#!/usr/bin/env bash
# Qwen3 1.7B/4B × no/visible-reasoning × Full/Reuse/RelayCaching/CacheBlend/EPIC/KVCOMM.
# Uses fixed format-switch 200 input. Jobs are intentionally phased: KVCOMM reads Full.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
INPUT="$ROOT/data/benchmark/benchmark_synthetic_format_switch_200/all.jsonl"
OUT="${OUT:-$ROOT/results/format_switch_200_baselines_$(date +%Y%m%d_%H%M%S)}"
PY="$ROOT/.venv_kvreuse/bin/python"
RELAY_PY="$ROOT/.venv_relaycaching/bin/python"
mkdir -p "$OUT"
export MODELSCOPE_CACHE="$ROOT/.modelscope" CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
# One 4B process/card; two 1.7B processes/card only when the scheduler starts them.
run_full() {
  local gpu=$1 model=$2 mode=$3 max=$4
  local extra=()
  [[ "$mode" == reasoning ]] && extra+=(--explicit-reasoning)
  env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/run_direct_reuse.py --model "$model" --input "$INPUT" --method full --boxed-output --no-plots --max-new-tokens "$max" "${extra[@]}" --output-root "$OUT/$mode/full" > "$OUT/${model}_${mode}_full.log" 2>&1
}
# Full artifacts first; run 1.7B/no, 4B/no, 1.7B/reasoning concurrently, then 4B/reasoning.
run_full 1 1.7b no_reasoning 128 & a=$!
run_full 2 4b no_reasoning 128 & b=$!
run_full 3 1.7b reasoning 512 & c=$!
wait "$a"; wait "$b"; wait "$c"
run_full 1 4b reasoning 512
run_method() {
  local gpu=$1 model=$2 mode=$3 method=$4 max=$5
  local extra=()
  [[ "$mode" == reasoning ]] && extra+=(--explicit-reasoning)
  case "$method" in
    reuse)
      env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/run_direct_reuse.py \
        --model "$model" --input "$INPUT" --method reuse --boxed-output --no-plots \
        --max-new-tokens "$max" "${extra[@]}" --output-root "$OUT/$mode/reuse"
      ;;
    kvcomm)
      env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/run_kvcomm.py \
        --model "$model" --input "$INPUT" \
        --baseline "$OUT/$mode/full/qwen3-$model/samples.jsonl" \
        --output-root "$OUT/$mode/kvcomm" --max-new-tokens "$max" "${extra[@]}"
      ;;
    relaycaching|cacheblend|epic)
      env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/third_party/RelayCaching" \
        "$RELAY_PY" -u scripts/run_relay_methods.py --method "$method" --model "$model" \
        --input "$INPUT" --output-root "$OUT/$mode" --max-new-tokens "$max" "${extra[@]}"
      ;;
  esac
}

jobs=(
  '1.7b no_reasoning reuse 128' '1.7b no_reasoning kvcomm 128' '1.7b no_reasoning relaycaching 128' '1.7b no_reasoning cacheblend 128' '1.7b no_reasoning epic 128'
  '4b no_reasoning reuse 128' '4b no_reasoning kvcomm 128' '4b no_reasoning relaycaching 128' '4b no_reasoning cacheblend 128' '4b no_reasoning epic 128'
  '1.7b reasoning reuse 512' '1.7b reasoning kvcomm 512' '1.7b reasoning relaycaching 512' '1.7b reasoning cacheblend 512' '1.7b reasoning epic 512'
  '4b reasoning reuse 512' '4b reasoning kvcomm 512' '4b reasoning relaycaching 512' '4b reasoning cacheblend 512' '4b reasoning epic 512'
)
worker() {
  local gpu=$1 slot=$2 job model mode method max
  for ((i=slot; i<${#jobs[@]}; i+=3)); do
    read -r model mode method max <<< "${jobs[i]}"
    echo "[GPU $gpu] $model $mode $method"
    run_method "$gpu" "$model" "$mode" "$method" "$max" > "$OUT/${model}_${mode}_${method}.log" 2>&1
  done
}
printf 'Full references complete; dispatching %s remaining method jobs: %s\n' "${#jobs[@]}" "$OUT" | tee "$OUT/STATUS.txt"
worker 1 0 & a=$!
worker 2 1 & b=$!
worker 3 2 & c=$!
wait "$a"; wait "$b"; wait "$c"
printf 'All baseline jobs complete: %s\n' "$OUT" | tee -a "$OUT/STATUS.txt"
