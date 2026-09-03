#!/usr/bin/env python3
"""Classify 4B reasoning failures on ArgKP and Deal by reuse failure mode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "reuse", "clean_reuse", "tail16_recompute", "tail16_post_recompute", "ours_post",
    "kvcomm", "relaycaching", "cacheblend", "epic",
)
DATASETS = ("argkp", "deal_or_no_deal")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def flatten(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize all saved schemas to one output row per target direction."""
    result = []
    for row in rows:
        if row.get("dataset") not in DATASETS:
            continue
        shared = {"task_id": row["task_id"], "dataset": row["dataset"]}
        if method == "full":
            for side, value in row["full"].items():
                result.append({**shared, "source": side, "target": side, **value})
        elif method == "kvcomm":
            for direction, value in row["calibration"].items():
                source, target = ("a", "b") if direction == "a_to_b" else ("b", "a")
                result.append({**shared, "source": source, "target": target, **value})
        elif row.get("reuse"):
            for direction, value in row["reuse"].items():
                source, target = ("a", "b") if direction == "a_to_b" else ("b", "a")
                result.append({**shared, "source": source, "target": target, **value})
        elif row.get("source_side") and row.get("target_side"):
            result.append(
                {
                    **shared,
                    "source": row["source_side"],
                    "target": row["target_side"],
                    "prediction": row.get("prediction"),
                    "gold": row.get("gold"),
                    "correct": row.get("correct", False),
                    "output_text": row.get("output_with_prefill", row.get("output", "")),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--paper-samples-dir", type=Path,
        default=Path("/tmp/kvreuse-4b-failure-samples"),
        help="Directory containing relaycaching.jsonl, epic.jsonl, and kvcomm.jsonl fetched from Lq-Sakura.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "analysis/outputs/4b_reasoning_argkp_deal_failure_modes.csv",
    )
    args = parser.parse_args()
    benchmark = args.results_root / "benchmark_argkp_deal_harmbench_301" / "reasoning"
    paths = {
        method: benchmark / method / "qwen3-4b" / "samples.jsonl" for method in METHODS
    }
    paths["relaycaching"] = args.paper_samples_dir / "relaycaching.jsonl"
    paths["epic"] = args.paper_samples_dir / "epic.jsonl"
    paths["kvcomm"] = args.paper_samples_dir / "kvcomm.jsonl"
    full_rows = flatten("full", read_rows(benchmark / "full" / "qwen3-4b" / "samples.jsonl"))
    full = {(row["task_id"], row["target"]): row for row in full_rows}

    report = []
    for dataset in DATASETS:
        for method in METHODS:
            rows = [row for row in flatten(method, read_rows(paths[method])) if row["dataset"] == dataset]
            failures = [row for row in rows if not row.get("correct", False)]
            source_match = [
                row for row in failures
                if row.get("prediction") == full[(row["task_id"], row["source"])].get("prediction")
                and row.get("prediction") != row.get("gold")
            ]
            source_gold = [
                row for row in source_match
                if full[(row["task_id"], row["source"])].get("correct", False)
            ]
            inherited = [
                row for row in failures
                if row not in source_match
                and not full[(row["task_id"], row["target"])].get("correct", False)
                and row.get("prediction") == full[(row["task_id"], row["target"])].get("prediction")
            ]
            unparsable = [row for row in failures if not row.get("prediction")]
            report.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "evaluated_directions": len(rows),
                    "failures": len(failures),
                    "source_full_answer_match": len(source_match),
                    "source_gold_leak": len(source_gold),
                    "inherited_target_full_error": len(inherited),
                    "other_target_error": len(failures) - len(source_match) - len(inherited),
                    "unparsed": len(unparsable),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
