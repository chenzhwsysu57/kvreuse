"""Unified schema and validation for constructed experiment examples."""

from __future__ import annotations

import hashlib
import json
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

OUTPUT_PROTOCOLS = {
    "option_letter", "raw_payload", "final_payload", "json_answer", "bracket_payload",
}


CORE_FIELDS = tuple(key for key in REQUIRED_FIELDS if key not in {"task_id", "metric"})


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy and validate an input, deriving a path-independent ID if absent.

    Only core text and an explicitly supplied output protocol contribute to
    generated IDs. Legacy IDs and additional fields are preserved unchanged.
    """
    validate_record(record)
    normalized = dict(record)
    if "task_id" not in normalized:
        payload = {key: record[key] for key in CORE_FIELDS}
        if "output_protocol" in record:
            payload["output_protocol"] = record["output_protocol"]
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
        normalized["task_id"] = "sample-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    normalized.setdefault("metric", "exact_match")
    return normalized


def validate_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("record must be a JSON object")
    missing = [key for key in CORE_FIELDS if key not in record]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for key in REQUIRED_FIELDS:
        if key not in record:
            continue
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if record["gold_a"] == record["gold_b"]:
        raise ValueError("gold_a and gold_b must differ")
    if record.get("metric", "exact_match") != "exact_match":
        raise ValueError("the MVP only supports exact_match")
    if "output_protocol" in record:
        protocol = record["output_protocol"]
        if not isinstance(protocol, str) or protocol not in OUTPUT_PROTOCOLS:
            raise ValueError(f"unsupported output_protocol: {protocol}")
