#!/usr/bin/env bash
# Remaining 8B reasoning 266 tasks for Lq-Sakura GPU 0.
set -euo pipefail

ROOT="${KVREUSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"
cd "$ROOT"

for method in reuse clean_reuse tail16_recompute; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 8b --reasoning on --method "$method" --overwrite "${RUNTIME_ARGS[@]}"
done
