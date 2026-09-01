#!/usr/bin/env bash
# Evaluate HelpSteer2 and PKU-SafeRLHF on the balanced 266-record benchmark.
# Example: bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh --model-size 4b --reasoning off
set -euo pipefail

usage() {
  echo "Usage: $0 --model-size {0.6b|1.7b|4b|8b} --reasoning {on|off} [--method METHOD] [--input PATH] [--output-root PATH] [--limit N] [--overwrite] [--dry-run] [--kvreuse-python PATH] [--relay-python PATH]" >&2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SIZE=""
REASONING=""
METHOD="full"
INPUT=""
OUTPUT_ROOT=""
LIMIT=""
OVERWRITE=0
DRY_RUN=0
KVREUSE_PYTHON="${KVREUSE_PYTHON:-python}"
RELAY_PYTHON="${RELAY_PYTHON:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-size) MODEL_SIZE="${2:-}"; shift 2 ;;
    --reasoning) REASONING="${2:-}"; shift 2 ;;
    --method) METHOD="${2:-}"; shift 2 ;;
    --input) INPUT="${2:-}"; shift 2 ;;
    --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --limit) LIMIT="${2:-}"; shift 2 ;;
    --kvreuse-python) KVREUSE_PYTHON="${2:-}"; shift 2 ;;
    --relay-python) RELAY_PYTHON="${2:-}"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$MODEL_SIZE" in 0.6b|1.7b|4b|8b) ;; *) usage; exit 2 ;; esac
case "$REASONING" in on) REASONING_ARG=yes ;; off) REASONING_ARG=no ;; *) usage; exit 2 ;; esac

INPUT="${INPUT:-$ROOT/data/benchmark/benchmark_helpsteer2_pku_safe_rlhf_266.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/helpsteer2_pku_safe_rlhf_266}"
[[ -f "$INPUT" ]] || { echo "Benchmark input not found: $INPUT" >&2; exit 1; }

COMMAND=(
  "$KVREUSE_PYTHON" -u "$ROOT/scripts/run_benchmark.py"
  --method "$METHOD"
  --reasoning "$REASONING_ARG"
  --model "$MODEL_SIZE"
  --input "$INPUT"
  --output-root "$OUTPUT_ROOT"
  --kvreuse-python "$KVREUSE_PYTHON"
)
[[ -n "$LIMIT" ]] && COMMAND+=(--limit "$LIMIT")
[[ "$OVERWRITE" -eq 1 ]] && COMMAND+=(--overwrite)
[[ "$DRY_RUN" -eq 1 ]] && COMMAND+=(--dry-run)
[[ -n "$RELAY_PYTHON" ]] && COMMAND+=(--relay-python "$RELAY_PYTHON")

cd "$ROOT"
exec "${COMMAND[@]}"
