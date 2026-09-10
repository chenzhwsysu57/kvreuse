#!/usr/bin/env bash
# Continuously occupy GPUs 1/2/3 with outstanding format-switch baselines.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
RUN="${RUN_DIR:-$ROOT/results/format_switch_200_baselines_20260908_135308}"
INPUT="$ROOT/data/benchmark/benchmark_synthetic_format_switch_200/all.jsonl"
PY="$ROOT/.venv_kvreuse/bin/python"
RELAY_PY="$ROOT/.venv_relaycaching/bin/python"
export MODELSCOPE_CACHE="$ROOT/.modelscope"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN/queue_logs"

run_direct() {
  local gpu="$1" model="$2" mode="$3" kind="$4" output="$5"
  shift 5
  [[ -f "$output" ]] && return 0
  mkdir -p "$(dirname "$output")"
  local args=(--input "$INPUT" --output "$output" --model "$model" --method "$kind" --batch-size 16)
  if [[ "$mode" == reasoning ]]; then
    args+=(--max-new-tokens 512 --explicit-reasoning)
  else
    args+=(--max-new-tokens 128)
  fi
  [[ "$kind" == reuse ]] && args+=(--reuse-engine scatter)
  echo "[$(date -Is)] GPU $gpu direct $model/$mode/$kind"
  env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u kv_semantic_attack/eval_synthetic_batch.py "${args[@]}"
}

run_relay() {
  local gpu="$1" model="$2" mode="$3" method="$4"
  local base="$RUN/relay"
  local summary="$base/$method/qwen3-$model/summary.json"
  [[ "$mode" == reasoning ]] && summary="$base/reasoning/$method/qwen3-$model/summary.json"
  [[ -f "$summary" ]] && return 0
  echo "[$(date -Is)] GPU $gpu relay $model/$mode/$method"
  local args=(--method "$method" --model "$model" --input "$INPUT" --output-root "$base" --max-new-tokens 128)
  [[ "$mode" == reasoning ]] && args+=(--explicit-reasoning --max-new-tokens 512)
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/third_party/RelayCaching" "$RELAY_PY" -u scripts/run_relay_methods.py "${args[@]}"
}

run_kvcomm() {
  local gpu="$1" model="$2" mode="$3"
  local output="$RUN/$mode/kvcomm/qwen3-$model/summary.json"
  [[ -f "$output" ]] && return 0
  local baseline="$RUN/$mode/full/qwen3-$model/samples.jsonl"
  [[ -f "$baseline" ]] || { echo "Missing Full baseline: $baseline" >&2; return 1; }
  echo "[$(date -Is)] GPU $gpu kvcomm $model/$mode"
  local args=(--model "$model" --input "$INPUT" --baseline "$baseline" --output-root "$RUN/$mode/kvcomm" --max-new-tokens 128)
  [[ "$mode" == reasoning ]] && args+=(--explicit-reasoning --max-new-tokens 512)
  env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/run_kvcomm.py "${args[@]}"
}

worker() {
  local gpu="$1"; shift
  while (( $# )); do
    "$@"
    return 0
  done
}

# Each worker consumes its list sequentially, automatically starting the next
# item immediately after the previous process exits.  All 1.7B tasks run first
# while the already-running 4B Full/reasoning job occupies GPU 1.
gpu1() {
  run_relay 1 1.7b no_reasoning cacheblend
  run_relay 1 1.7b no_reasoning epic
  run_direct 1 1.7b reasoning reuse "$RUN/reasoning/reuse_1p7b_batch16.json"
  run_relay 1 1.7b reasoning relaycaching
  run_kvcomm 1 1.7b no_reasoning
  run_kvcomm 1 1.7b reasoning
  run_direct 1 4b no_reasoning reuse "$RUN/no_reasoning/reuse_4b_batch16.json"
  run_relay 1 4b no_reasoning relaycaching
  run_relay 1 4b no_reasoning cacheblend
  run_relay 1 4b no_reasoning epic
  run_kvcomm 1 4b no_reasoning
  run_direct 1 4b reasoning reuse "$RUN/reasoning/reuse_4b_batch16.json"
  run_relay 1 4b reasoning relaycaching
  run_relay 1 4b reasoning cacheblend
  run_relay 1 4b reasoning epic
  run_kvcomm 1 4b reasoning
}

gpu2() {
  run_direct 2 1.7b no_reasoning reuse "$RUN/no_reasoning/reuse_1p7b_batch16.json"
  run_relay 2 1.7b no_reasoning epic
  run_relay 2 1.7b reasoning relaycaching
  run_relay 2 1.7b reasoning cacheblend
  run_relay 2 1.7b reasoning epic
  run_kvcomm 2 1.7b no_reasoning
  run_kvcomm 2 1.7b reasoning
}

gpu3() {
  run_direct 3 1.7b reasoning reuse "$RUN/reasoning/reuse_1p7b_batch16.json"
  run_relay 3 1.7b reasoning cacheblend
  run_relay 3 1.7b reasoning epic
  run_kvcomm 3 1.7b no_reasoning
}

case "${1:-}" in
  1) gpu1 ;;
  2) gpu2 ;;
  3) gpu3 ;;
  *) echo "usage: $0 {1|2|3}" >&2; exit 2 ;;
esac