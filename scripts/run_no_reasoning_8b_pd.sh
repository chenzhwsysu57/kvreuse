#!/usr/bin/env bash
# Run every current PD entry in results/no-reasoning-qwen3-8b.csv.
# The script intentionally excludes ours_precaution, matching the progress CSVs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SIZE="8b"
REASONING="off"
source "$ROOT/scripts/batch_runtime_args.sh"
parse_batch_runtime_args "$@"

cd "$ROOT"

# All ten tracked methods are PD for HelpSteer2 / PKU-SafeRLHF at 8B.
# Full runs first because KVCOMM requires its completed local artifact.  The
# 301 methods have already completed and are deliberately excluded.
for method in full reuse clean_reuse tail16_recompute tail16_post_recompute ours_post relaycaching cacheblend epic kvcomm; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size "$MODEL_SIZE" --reasoning "$REASONING" --method "$method" --overwrite "${RUNTIME_ARGS[@]}"
done

echo "Completed local runs. Refresh the CSV with the Lq-Sakura remote-result overlays as well."
