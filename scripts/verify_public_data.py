#!/usr/bin/env python3
"""Verify release data integrity and family-level train/test isolation (CPU only)."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kvreuse_data.schema import validate_record


def verify() -> dict[str, int]:
    data = ROOT / "data"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    rows_by_file = {}
    actual_files = {p.relative_to(data).as_posix() for p in data.rglob("*.jsonl")}
    if actual_files != set(manifest["files"]):
        raise ValueError("unexpected or missing release data files")
    for name, spec in manifest["files"].items():
        path = data / name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {name}")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != spec["records"]:
            raise ValueError(f"record-count mismatch: {name}")
        for row in rows:
            errors = validate_record(row)
            if errors:
                raise ValueError(f"invalid record in {name}: {errors}")
        if "families" in spec:
            counts = Counter(row["dataset"] for row in rows)
            if len(counts) != spec["families"] or set(counts.values()) != {30}:
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