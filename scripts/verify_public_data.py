#!/usr/bin/env python3
"""Verify compact release data and family isolation without a manifest."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kvreuse_data.schema import validate_record

EXPECTED = {
    "benchmark/benchmark_argkp_deal_220.jsonl": (220, None),
    "benchmark/benchmark_job_interview_110.jsonl": (110, None),
    "benchmark/benchmark_final240_real_ood_330_20261006.jsonl": (330, None),
    "benchmark/generated240/full_240classes_30each.jsonl": (7200, 240),
    "benchmark/generated240/train_180classes_30each.jsonl": (5400, 180),
    "benchmark/generated240/test_60classes_30each.jsonl": (1800, 60),
}
CORE = {"dataset", "prefix_a", "prefix_b", "shared_block", "question", "gold_a", "gold_b"}


def verify() -> dict[str, int]:
    data = ROOT / "data"
    rows_by_file = {}
    actual_files = {p.relative_to(data).as_posix() for p in data.rglob("*.jsonl")}
    if actual_files != set(EXPECTED):
        raise ValueError("unexpected or missing release data files")
    if list(data.rglob("*manifest*.json")):
        raise ValueError("release data must not contain manifests")
    for name, (count, family_count) in EXPECTED.items():
        path = data / name
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != count:
            raise ValueError(f"record-count mismatch: {name}")
        for row in rows:
            if not CORE <= row.keys() or set(row) - CORE - {"output_protocol"}:
                raise ValueError(f"non-minimal input fields: {name}")
            validate_record(row)
        if family_count is not None:
            counts = Counter(row["dataset"] for row in rows)
            if len(counts) != family_count or set(counts.values()) != {30}:
                raise ValueError(f"family-count mismatch: {name}")
        rows_by_file[name] = rows
    prefix = "benchmark/generated240/"
    full = rows_by_file[prefix + "full_240classes_30each.jsonl"]
    train = rows_by_file[prefix + "train_180classes_30each.jsonl"]
    test = rows_by_file[prefix + "test_60classes_30each.jsonl"]
    families = lambda rows: {row["dataset"] for row in rows}
    if families(train) & families(test):
        raise ValueError("training and held-out families overlap")
    fingerprint = lambda rows: Counter(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
    if fingerprint(full) != fingerprint(train + test):
        raise ValueError("full corpus is not exactly the train/test union")
    if families(full) != {f"family_{index:03d}" for index in range(1, 241)}:
        raise ValueError("synthetic family names must be compact and anonymous")
    combined = rows_by_file["benchmark/benchmark_final240_real_ood_330_20261006.jsonl"]
    source = rows_by_file["benchmark/benchmark_argkp_deal_220.jsonl"] + rows_by_file["benchmark/benchmark_job_interview_110.jsonl"]
    if fingerprint(source) != fingerprint(combined):
        raise ValueError("combined real-world data differs from its source slices")
    if Counter(row["dataset"] for row in combined) != {"argkp": 110, "deal_or_no_deal": 110, "job_interview": 110}:
        raise ValueError("unexpected real-world evaluation datasets")
    return {"files": len(rows_by_file), "full_families": len(families(full)),
            "train_families": len(families(train)), "test_families": len(families(test)),
            "real_world_pairs": len(combined)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))