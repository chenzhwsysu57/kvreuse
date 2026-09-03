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


def validate_job_interview_record(record: dict[str, object]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("JobInterview metadata is required")
    candidates = metadata.get("candidate_offers")
    worker_scores = metadata.get("worker_scores")
    recruiter_scores = metadata.get("recruiter_scores")
    if not isinstance(candidates, list) or not 3 <= len(candidates) <= 8:
        raise ValueError("JobInterview must contain three to eight candidate contracts")
    if not isinstance(worker_scores, list) or not isinstance(recruiter_scores, list):
        raise ValueError("JobInterview utility scores are required")
    if len(worker_scores) != len(candidates) or len(recruiter_scores) != len(candidates):
        raise ValueError("JobInterview candidate/score length mismatch")
    required_terms = {"Salary", "Position", "Weekly holiday", "Workplace", "Company"}
    if any(not isinstance(offer, dict) or set(offer) != required_terms for offer in candidates):
        raise ValueError("JobInterview candidate contract has invalid terms")
    try:
        worker_values = [float(value) for value in worker_scores]
        recruiter_values = [float(value) for value in recruiter_scores]
        margin = float(metadata.get("minimum_margin"))
    except (TypeError, ValueError):
        raise ValueError("JobInterview scores and margin must be numeric") from None
    worker_best = [index for index, value in enumerate(worker_values) if abs(value - max(worker_values)) < 1e-12]
    recruiter_best = [index for index, value in enumerate(recruiter_values) if abs(value - max(recruiter_values)) < 1e-12]
    if len(worker_best) != 1 or len(recruiter_best) != 1 or worker_best == recruiter_best:
        raise ValueError("JobInterview requires distinct unique role optima")
    if record["gold_a"] != chr(ord("A") + worker_best[0]):
        raise ValueError("JobInterview worker gold does not select the utility optimum")
    if record["gold_b"] != chr(ord("A") + recruiter_best[0]):
        raise ValueError("JobInterview recruiter gold does not select the utility optimum")
    if max(worker_values) - sorted(worker_values)[-2] + 1e-12 < margin:
        raise ValueError("JobInterview worker utility margin is below threshold")
    if max(recruiter_values) - sorted(recruiter_values)[-2] + 1e-12 < margin:
        raise ValueError("JobInterview recruiter utility margin is below threshold")


def validate_fantom_record(record: dict[str, object]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("FANToM metadata is required")
    joining = metadata.get("joining_speaker")
    witness = metadata.get("witness")
    known = metadata.get("known_speakers")
    order = metadata.get("candidate_order")
    kinds = metadata.get("candidate_types")
    if not isinstance(joining, str) or not joining or not isinstance(witness, str) or not witness:
        raise ValueError("FANToM speaker identities are required")
    if not isinstance(known, list) or witness not in known or joining in known:
        raise ValueError("FANToM information-access invariant failed")
    if metadata.get("information_access") != "joining_speaker_inaccessible_vs_witness_accessible":
        raise ValueError("FANToM access construction is invalid")
    if not isinstance(order, list) or sorted(order) != [0, 1]:
        raise ValueError("FANToM candidate order is invalid")
    if not isinstance(kinds, list) or len(kinds) != 2 or set(kinds) != {"full_information", "limited_information"}:
        raise ValueError("FANToM candidate types are invalid")
    full_index = kinds.index("full_information")
    limited_index = kinds.index("limited_information")
    if record["gold_a"] != chr(ord("A") + limited_index):
        raise ValueError("FANToM joining-speaker gold must select limited information")
    if record["gold_b"] != chr(ord("A") + full_index):
        raise ValueError("FANToM witness gold must select full information")
    shared = str(record["shared_block"])
    if not shared.startswith("Conversation:\n") or "\n\nRelevant missed information:\n" not in shared:
        raise ValueError("FANToM shared context markers are missing")


def validate_fantom_access_record(record: dict[str, object]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("FANToM-access metadata is required")
    joining = metadata.get("joining_speaker")
    witness = metadata.get("witness")
    known = metadata.get("known_speakers")
    answer_order = metadata.get("answer_order")
    if not isinstance(joining, str) or not joining or not isinstance(witness, str) or not witness:
        raise ValueError("FANToM-access speaker identities are required")
    if not isinstance(known, list) or witness not in known or joining in known:
        raise ValueError("FANToM-access information invariant failed")
    if metadata.get("information_access") != "joining_speaker_inaccessible_vs_witness_accessible":
        raise ValueError("FANToM-access construction is invalid")
    if not isinstance(answer_order, list) or sorted(answer_order) != ["NO", "YES"]:
        raise ValueError("FANToM-access answer order is invalid")
    if record["gold_a"] != chr(ord("A") + answer_order.index("NO")):
        raise ValueError("FANToM-access joining-speaker gold must be NO")
    if record["gold_b"] != chr(ord("A") + answer_order.index("YES")):
        raise ValueError("FANToM-access witness gold must be YES")
    shared = str(record["shared_block"])
    if not shared.startswith("Conversation:\n") or "\n\nProposition:\n" not in shared or "\n\nAnswer options:\n" not in shared:
        raise ValueError("FANToM-access shared context markers are missing")


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
                    elif record["dataset"] == "job_interview":
                        validate_job_interview_record(record)
                    elif record["dataset"] == "fantom":
                        validate_fantom_record(record)
                    elif record["dataset"] == "fantom_access":
                        validate_fantom_access_record(record)
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
