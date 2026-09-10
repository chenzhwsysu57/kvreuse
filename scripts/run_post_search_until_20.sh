#!/usr/bin/env bash
# Search universal post wording until it reaches 20% direct-reuse accuracy.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
PY="${PYTHON:-$ROOT/.venv_kvreuse/bin/python}"
INPUT="${INPUT:-$ROOT/kv_semantic_attack/adaptive_runs/local_20260907_151934/round_01_candidates.jsonl}"
SEED_DASHBOARD="${SEED_DASHBOARD:-$ROOT/kv_semantic_attack/adaptive_runs/defense5_gpu123_20260907_161449/initial_dashboard.json}"
RUN="${RUN_DIR:-$ROOT/kv_semantic_attack/adaptive_runs/post_search_until20_$(date +%Y%m%d_%H%M%S)}"
[[ "$RUN" = /* ]] || RUN="$ROOT/$RUN"
[[ -x "$PY" && -f "$INPUT" && -f "$SEED_DASHBOARD" ]] || { echo "Missing interpreter, input, or seed dashboard." >&2; exit 1; }
mkdir "$RUN"
exec > >(tee -a "$RUN/pipeline.log") 2>&1
export MODELSCOPE_CACHE="$ROOT/.modelscope"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
unset CUDA_VISIBLE_DEVICES
# The inherited ALL_PROXY caused the adaptive API to time out; direct access was verified.
if [[ "${KEEP_PROXY:-0}" != 1 ]]; then
  unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
fi
cp "$SEED_DASHBOARD" "$RUN/seed_dashboard.json"
echo "Run: $RUN"
echo "Target: direct reuse + universal post >= 20.0%; Qwen3-1.7B, no-reasoning, boxed, batch=8"
echo "Search: up to 100 rounds; 4 new candidates per round; GPUs 1/2/3, two slots each."
"$PY" -u kv_semantic_attack/run_defense_refinement.py \
  --input "$INPUT" --dashboard "$RUN/seed_dashboard.json" --run-dir "$RUN/refinement" \
  --start-step 1 --max-attempts 100 --candidates-per-attempt 4 \
  --target-accuracy 0.20 --minimum-improvement-pp 3 \
  --model 1.7b --batch-size 8 --max-new-tokens 128 \
  --gpu-slots 1=2 2=2 3=2 --max-used-gib 4