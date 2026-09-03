#!/usr/bin/env python3
"""Audit percentage-valued progress-board cells against saved sample artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BENCHMARKS = {
    "argkp-110": ("benchmark_argkp_deal_harmbench_301", "argkp", 110),
    "deal_or_no_deal-110": ("benchmark_argkp_deal_harmbench_301", "deal_or_no_deal", 110),
    "harmbench_contextual-81": ("benchmark_argkp_deal_harmbench_301", "harmbench_contextual", 81),
    "helpsteer2-133": ("helpsteer2_pku_safe_rlhf_266", "helpsteer2", 133),
    "pku_safe_rlhf-133": ("helpsteer2_pku_safe_rlhf_266", "pku_safe_rlhf", 133),
}
BOARD_NAMES = (
    "done-no-reasoning-qwen3-1.7b.csv",
    "done-no-reasoning-qwen3-4b.csv",
    "no-reasoning-qwen3-8b.csv",
    "reasoning-qwen3-1.7b.csv",
    "reasoning-qwen3-4b.csv",
    "reasoning-qwen3-8b.csv",
)


def successful_direction_keys(rows: list[dict[str, object]], method: str) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            continue
        if method == "kvcomm":
            keys.update(
                (task_id, direction)
                for direction, value in row.get("calibration", {}).items()
                if value.get("correct") is not None
            )
        elif method == "full" and row.get("full"):
            keys.update(
                (task_id, direction)
                for direction, value in row["full"].items()
                if value.get("correct") is not None
            )
        elif row.get("reuse"):
            keys.update(
                (task_id, direction)
                for direction, value in row["reuse"].items()
                if value.get("correct") is not None
            )
        elif row.get("correct") is not None and row.get("source_side") and row.get("target_side"):
            keys.add((task_id, f"{row['source_side']}_to_{row['target_side']}"))
    return keys


def audit_cell(results_root: Path, row: dict[str, str]) -> dict[str, str]:
    benchmark, artifact_dataset, records = BENCHMARKS[row["dataset"]]
    mode_dir = "reasoning" if row["reasoning"] == "on" else "no_reasoning"
    samples = results_root / benchmark / mode_dir / row["method"] / row["model"] / "samples.jsonl"
    expected = records * 2
    base = {
        "board": row["board"], "dataset": row["dataset"], "model": row["model"],
        "method": row["method"], "reasoning": row["reasoning"], "accuracy": row["status"],
        "expected_directions": str(expected), "samples_path": str(samples),
    }
    if not samples.is_file():
        return {**base, "artifact_state": "missing", "successful_directions": "0"}
    rows = [
        json.loads(line)
        for line in samples.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line).get("dataset") == artifact_dataset
    ]
    successful = len(successful_direction_keys(rows, row["method"]))
    state = "complete" if successful == expected else "partial"
    return {**base, "artifact_state": state, "successful_directions": str(successful)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards-dir", type=Path, default=Path("results"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--secondary-audit",
        type=Path,
        help="Optional audit CSV from another machine; merge each cell by its most complete artifact.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit_rows = []
    for board_name in BOARD_NAMES:
        board = args.boards_dir / board_name
        if not board.is_file():
            continue
        with board.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["status"].endswith("%"):
                    audit_rows.append(audit_cell(args.results_root, {**row, "board": board_name}))

    fields = (
        "board", "dataset", "model", "method", "reasoning", "accuracy",
        "artifact_state", "successful_directions", "expected_directions", "samples_path",
    )
    if args.secondary_audit:
        with args.secondary_audit.open(newline="") as handle:
            remote_rows = list(csv.DictReader(handle))
        keyed = {}
        for row in audit_rows + remote_rows:
            key = (row["board"], row["dataset"], row["model"], row["method"], row["reasoning"])
            previous = keyed.get(key)
            priority = {"missing": 0, "partial": 1, "complete": 2}
            if previous is None or (priority[row["artifact_state"]], int(row["successful_directions"])) > (
                priority[previous["artifact_state"]], int(previous["successful_directions"])
            ):
                keyed[key] = row
        audit_rows = list(keyed.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"wrote {args.output} ({len(audit_rows)} percentage cells)")


if __name__ == "__main__":
    main()
