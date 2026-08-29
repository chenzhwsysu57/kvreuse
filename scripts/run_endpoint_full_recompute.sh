#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

result_dir="results/full_recompute_validation_harmbench"
validation_file="data/validation/second_step_10_each_harmbench.jsonl"

# Fetch the three official PKU-SafeRLHF test shards and pinned HarmBench test files.
# Cached files are reused.
conda run -n kvreuse python scripts/download_datasets.py \
  --datasets pku_safe_rlhf harmbench_contextual

# Rebuild all five datasets. HelpSteer2 uses {0, 4}; PKU uses strict
# helpfulness-vs-safety conflicts; HarmBench uses contextual intent flips.
conda run -n kvreuse python scripts/build_datasets.py \
  --seed 20260827 \
  --max-per-dataset 1000
conda run -n kvreuse python scripts/validate_datasets.py
conda run -n kvreuse python scripts/select_validation_subset.py \
  --input data/processed/all.jsonl \
  --output "$validation_file" \
  --per-dataset 10 \
  --seed 20260827

# Use a fresh result root so five-dataset runs cannot mix with older reports.
for model_size in 0.6b 1.7b 4b 8b; do
  conda run -n kvreuse python scripts/run_full_recompute_validation.py \
    --model "$model_size" \
    --input "$validation_file" \
    --output-root "$result_dir" \
    --retry-truncated \
    --retry-max-new-tokens 1024 \
    --allow-download
done

conda run -n kvreuse python scripts/summarize_full_recompute_validation.py \
  --root "$result_dir"
