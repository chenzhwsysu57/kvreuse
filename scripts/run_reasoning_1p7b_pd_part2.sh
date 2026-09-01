#!/usr/bin/env bash
# PD batch 2: independent 266 bridge and RelayCaching-family methods.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for method in tail16_post_recompute ours_post relaycaching cacheblend; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size 1.7b --reasoning on --method "$method"
done
