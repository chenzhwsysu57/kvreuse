#!/usr/bin/env python3
"""Build a deterministic, dataset-selectable KV-reuse benchmark.

The input is the already validated ``data/processed/all.jsonl``.  Sampling is
deterministic and happens independently per dataset, so rerunning this command
does not change a previously published benchmark unless the input or seed is
changed.  With no selection arguments, the historical five-dataset 250-record
benchmark (50 records per dataset) is reproduced exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUOTAS = {
    "argkp": 50,
    "deal_or_no_deal": 50,
    "harmbench_contextual": 50,
    "helpsteer2": 50,
    "pku_safe_rlhf": 50,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/all.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark/benchmark_250.jsonl")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--quotas-json", type=str,
        help="JSON mapping of dataset name to selected-record count. Overrides --datasets/--per-dataset.",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="datasets to include; use with --per-dataset, e.g. argkp deal_or_no_deal harmbench_contextual",
    )
    parser.add_argument(
        "--per-dataset", type=int,
        help="deterministic sample size for every --datasets entry",
    )
    parser.add_argument(
        "--allow-fewer", action="store_true",
        help="when a selected dataset has fewer than --per-dataset records, include all available records",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quotas_json is not None:
        if args.datasets or args.per_dataset is not None or args.allow_fewer:
            raise ValueError("--quotas-json cannot be combined with --datasets, --per-dataset, or --allow-fewer")
        quotas = json.loads(args.quotas_json)
    elif args.datasets:
        if args.per_dataset is None or args.per_dataset <= 0:
            raise ValueError("--datasets requires a positive --per-dataset")
        if len(set(args.datasets)) != len(args.datasets):
            raise ValueError("--datasets must not contain duplicates")
        quotas = {dataset: args.per_dataset for dataset in args.datasets}
    else:
        if args.per_dataset is not None or args.allow_fewer:
            raise ValueError("--per-dataset and --allow-fewer require --datasets")
        quotas = dict(DEFAULT_QUOTAS)
    if not isinstance(quotas, dict) or not all(isinstance(value, int) and value > 0 for value in quotas.values()):
        raise ValueError("--quotas-json must map datasets to positive integer counts")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; use --overwrite to rebuild it")

    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in quotas}
    for row in load_jsonl(args.input):
        if row["dataset"] in grouped:
            grouped[row["dataset"]].append(row)
    unknown = sorted(set(quotas) - set(grouped))
    if unknown:
        raise ValueError(f"requested datasets absent from input: {unknown}")
    selected: list[dict[str, Any]] = []
    requested_quotas = dict(quotas)
    for dataset, quota in quotas.items():
        candidates = grouped[dataset]
        if len(candidates) < quota:
            if not args.allow_fewer:
                raise ValueError(
                    f"{dataset}: requested {quota}, but only {len(candidates)} records exist; "
                    "pass --allow-fewer to include all available records"
                )
            quota = len(candidates)
        # A dataset-specific RNG keeps a changed quota in one dataset from
        # perturbing the selected IDs in all other datasets.
        rng = random.Random(f"kvreuse-benchmark-v1:{args.seed}:{dataset}")
        selected.extend(rng.sample(candidates, quota))
    # A deterministic interleave prevents each dataset from occupying one long
    # contiguous range while retaining the exact per-dataset counts above.
    rng = random.Random(f"kvreuse-benchmark-v1:{args.seed}:interleave")
    rng.shuffle(selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
    manifest = {
        "name": f"kvreuse_benchmark_{len(selected)}_v1",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "seed": args.seed,
        "requested_quotas": requested_quotas,
        "quotas": dict(sorted(Counter(row["dataset"] for row in selected).items())),
        "counts": dict(sorted(Counter(row["dataset"] for row in selected).items())),
        "task_ids": [row["task_id"] for row in selected],
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(selected), "counts": manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
