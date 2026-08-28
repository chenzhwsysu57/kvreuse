#!/usr/bin/env python3
"""Validate schema, uniqueness, and the cross-prefix gold-change invariant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvreuse_data.schema import validate_record


def validate_pku_record(record: dict[str, object]) -> None:
    if record["gold_a"] not in {"A", "B"} or record["gold_b"] not in {"A", "B"}:
        raise ValueError("PKU gold labels must be A/B")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("strict_conflict") is not True:
        raise ValueError("PKU record is not marked as a strict conflict")
    flags = metadata.get("displayed_safe_flags")
    severities = metadata.get("displayed_severity_levels")
    if not isinstance(flags, list) or len(flags) != 2 or set(flags) != {False, True}:
        raise ValueError("PKU record must display exactly one safe response")
    if not isinstance(severities, list) or len(severities) != 2:
        raise ValueError("PKU record must include two severity levels")
    safe_index = flags.index(True)
    unsafe_index = flags.index(False)
    if record["gold_b"] != chr(ord("A") + safe_index):
        raise ValueError("PKU safety gold does not point to the safe response")
    if record["gold_a"] != chr(ord("A") + unsafe_index):
        raise ValueError("PKU helpfulness gold does not point to the conflicting unsafe response")
    if int(severities[safe_index]) != 0 or int(severities[unsafe_index]) < 2:
        raise ValueError("PKU severity filter invariant failed")
    if "[A] " not in str(record["shared_block"]) or "[B] " not in str(record["shared_block"]):
        raise ValueError("PKU shared block must contain displayed A/B responses")


def validate_harmbench_record(record: dict[str, object]) -> None:
    if record["gold_a"] != "BENIGN" or record["gold_b"] != "HARMFUL":
        raise ValueError("HarmBench intent gold must flip BENIGN -> HARMFUL")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("HarmBench metadata is required")
    if metadata.get("functional_category") != "contextual":
        raise ValueError("HarmBench record must come from the contextual category")
    tags = metadata.get("tags")
    if not isinstance(tags, list) or "context" not in tags:
        raise ValueError("HarmBench contextual record must carry the context tag")
    behavior = metadata.get("behavior")
    if not isinstance(behavior, str) or not behavior or behavior not in str(record["prefix_b"]):
        raise ValueError("HarmBench harmful prefix must contain the official behavior")
    shared = str(record["shared_block"])
    marker = "Material under consideration:\n"
    if not shared.startswith(marker):
        raise ValueError("HarmBench shared block marker is missing")
    import hashlib

    context = shared[len(marker):]
    if hashlib.sha256(context.encode("utf-8")).hexdigest() != metadata.get("context_sha256"):
        raise ValueError("HarmBench context hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("data/processed/all.jsonl")])
    args = parser.parse_args()
    seen: set[str] = set()
    total = 0
    for path in args.paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                try:
                    validate_record(record)
                    if record["dataset"] == "pku_safe_rlhf":
                        validate_pku_record(record)
                    elif record["dataset"] == "harmbench_contextual":
                        validate_harmbench_record(record)
                except ValueError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                if record["task_id"] in seen:
                    raise ValueError(f"duplicate task_id: {record['task_id']}")
                seen.add(record["task_id"])
                total += 1
    print(f"validated {total} records across {len(args.paths)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
