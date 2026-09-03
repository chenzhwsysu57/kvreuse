#!/usr/bin/env bash
# Remaining 8B reasoning 266 tasks for Lq-Sakura GPU 2.
set -euo pipefail

ROOT="${KVREUSE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"
cd "$ROOT"

# Full is a KVCOMM prerequisite only and is already tracked separately in CSV.
if [[ ! -f results/helpsteer2_pku_safe_rlhf_266/reasoning/full/qwen3-8b/samples.jsonl ]]; then
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 8b --reasoning on --method full --overwrite "${RUNTIME_ARGS[@]}"
fi

for method in kvcomm relaycaching; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 8b --reasoning on --method "$method" --overwrite "${RUNTIME_ARGS[@]}"
done
