#!/usr/bin/env bash
# Run the three RelayCaching-repository baselines under the static-block adapter.
# Usage: bash scripts/run_relay_methods_all.sh 1.7b [extra run_relay_methods.py args]
set -euo pipefail

model="${1:-1.7b}"
if [[ $# -gt 0 ]]; then
  shift
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/czw/miniconda3/envs/relaycaching/bin/python"
export PYTHONPATH="${root}/third_party/RelayCaching${PYTHONPATH:+:${PYTHONPATH}}"

for method in relaycaching cacheblend epic; do
  "${python_bin}" -u "${root}/scripts/run_relay_methods.py" \
    --method "${method}" --model "${model}" "$@"
done
