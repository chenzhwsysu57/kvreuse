"""Unified schema and validation for constructed experiment examples."""

from __future__ import annotations

from typing import Any, Mapping


REQUIRED_FIELDS = (
    "task_id",
    "dataset",
    "prefix_a",
    "prefix_b",
    "shared_block",
    "question",
    "gold_a",
    "gold_b",
    "metric",
)


def validate_record(record: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_FIELDS if key not in record]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for key in REQUIRED_FIELDS:
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if record["gold_a"] == record["gold_b"]:
        raise ValueError("gold_a and gold_b must differ")
    if record["metric"] != "exact_match":
        raise ValueError("the MVP only supports exact_match")
