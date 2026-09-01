#!/usr/bin/env bash
# Run every current PD entry in results/no-reasoning-qwen3-8b.csv.
# The script intentionally excludes ours_precaution, matching the progress CSVs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_SIZE="8b"
REASONING="off"
PYTHON_BIN="${KVREUSE_PYTHON:-python}"

cd "$ROOT"

# The 301 Full result is recorded from Lq-Sakura, but its sample artifact is
# not stored locally.  KVCOMM needs that local Full artifact for calibration.
if [[ ! -f results/benchmark_argkp_deal_harmbench_301/no_reasoning/full/qwen3-8b/samples.jsonl ]]; then
  "$PYTHON_BIN" -u scripts/run_benchmark.py \
    --method full --reasoning no --model "$MODEL_SIZE" \
    --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
    --output-root results/benchmark_argkp_deal_harmbench_301 \
    --kvreuse-python "$PYTHON_BIN"
fi

# Remaining PD methods on the 301-record ArgKP / Deal / HarmBench benchmark.
for method in relaycaching cacheblend epic kvcomm; do
  "$PYTHON_BIN" -u scripts/run_benchmark.py \
    --method "$method" --reasoning no --model "$MODEL_SIZE" \
    --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
    --output-root results/benchmark_argkp_deal_harmbench_301 \
    --kvreuse-python "$PYTHON_BIN"
done

# All ten tracked methods are still PD for HelpSteer2 / PKU-SafeRLHF at 8B.
# Full runs first because KVCOMM requires its completed local artifact.
for method in full reuse clean_reuse tail16_recompute tail16_post_recompute ours_post relaycaching cacheblend epic kvcomm; do
  bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh \
    --model-size "$MODEL_SIZE" --reasoning "$REASONING" --method "$method"
done

echo "Completed local runs. Refresh the CSV with the Lq-Sakura remote-result overlays as well."
