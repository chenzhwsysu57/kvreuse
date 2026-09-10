#!/usr/bin/env bash
# Re-score the fixed pool, then run five feedback-driven defense rounds.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
PY="${PYTHON:-$ROOT/.venv_kvreuse/bin/python}"
OLD="$ROOT/kv_semantic_attack/adaptive_runs/local_20260907_151934"
INPUT="${INPUT:-$OLD/round_01_candidates.jsonl}"
CANDIDATES="${CANDIDATES:-$OLD/round_01_suffixes.json}"
RUN="${RUN_DIR:-$ROOT/kv_semantic_attack/adaptive_runs/defense5_gpu123_$(date +%Y%m%d_%H%M%S)}"
[[ "$RUN" = /* ]] || RUN="$ROOT/$RUN"
[[ -x "$PY" && -f "$INPUT" && -f "$CANDIDATES" ]] || { echo "Missing interpreter/input/candidates" >&2; exit 1; }
# Never reuse a previous run directory.
mkdir "$RUN"
exec > >(tee -a "$RUN/pipeline.log") 2>&1
export MODELSCOPE_CACHE="$ROOT/.modelscope"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
unset CUDA_VISIBLE_DEVICES
# This machine's inherited ALL_PROXY timed out for the configured API.
# Direct API access was verified; opt back in explicitly when needed elsewhere.
if [[ "${KEEP_PROXY:-0}" != 1 ]]; then
  unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
fi
echo "Run directory: $RUN"
echo "GPU slots: 1=2 2=2 3=2; model=1.7b, no-reasoning, batch=8"

"$PY" -u kv_semantic_attack/run_multi_gpu_defense.py \
  --input "$INPUT" --candidates "$CANDIDATES" \
  --output-dir "$RUN/baseline_parallel" --output "$RUN/baseline_evaluation.json" \
  --gpu-slots 1=2 2=2 3=2 --max-used-gib 4 \
  --model 1.7b --batch-size 8 --max-new-tokens 128 \
  --run-dir "$RUN" --step 1

"$PY" kv_semantic_attack/build_adaptive_dashboard.py \
  --evaluation "$RUN/baseline_evaluation.json" --source "$INPUT" \
  --candidate direct_reuse --examples-per-type 6 --include-candidate-feedback --output "$RUN/initial_dashboard.json" \
  --run-dir "$RUN" --step 2

"$PY" -u kv_semantic_attack/run_defense_refinement.py \
  --input "$INPUT" --dashboard "$RUN/initial_dashboard.json" \
  --run-dir "$RUN/refinement" --start-step 1 --max-attempts 5 \
  --candidates-per-attempt 4 --minimum-improvement-pp 3 \
  --model 1.7b --batch-size 8 --max-new-tokens 128 \
  --gpu-slots 1=2 2=2 3=2 --max-used-gib 4

echo "All five rounds completed: $RUN/refinement/refinement_summary.json"