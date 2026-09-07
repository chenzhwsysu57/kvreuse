#!/usr/bin/env python3
"""Turn one defense evaluation into the next-round attack/defense dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kv_semantic_attack.run_journal import file_ref, write_step


def select_result(evaluation: dict, candidate_id: str) -> tuple[str, str, list[dict]]:
    if candidate_id == "direct_reuse":
        return "direct_reuse", "", evaluation["direct_reuse"]["results"]
    for entry in evaluation["candidates"]:
        candidate = entry["candidate"]
        if candidate["candidate_id"] == candidate_id:
            return candidate_id, candidate["suffix"], entry["results"]
    raise ValueError(f"unknown candidate_id: {candidate_id}")


def compact_example(row: dict, source: dict) -> dict:
    return {
        "task_id": row["task_id"], "target": row["target"],
        "source_gold": row["source_gold"], "target_gold": row["full"]["gold"],
        "full_prediction": row["full"]["prediction"],
        "reuse_prediction": row["reuse"]["prediction"],
        "prefix_target": source[f"prefix_{row['target']}"],
        "prefix_source": source[f"prefix_{'b' if row['target'] == 'a' else 'a'}"],
        "shared_block": source["shared_block"], "question": source["question"],
    }


def build_dashboard(evaluation: dict, sources: dict[str, dict], *, candidate_id: str,
                    examples_per_type: int) -> dict:
    selected_id, suffix, results = select_result(evaluation, candidate_id)
    groups = defaultdict(list)
    for row in results:
        groups[row["attack_type"]].append(row)
    by_type, failures, coverage = {}, {}, {}
    for attack_type, rows in sorted(groups.items()):
        full_correct = [row for row in rows if row["full"]["correct"]]
        valid_failures = [row for row in full_correct if not row["reuse"]["correct"]]
        leakage = sum(row["reuse"]["prediction"] == row["source_gold"] for row in valid_failures)
        by_type[attack_type] = {
            "directions": len(rows), "full_accuracy": sum(row["full"]["correct"] for row in rows) / len(rows),
            "reuse_accuracy": sum(row["reuse"]["correct"] for row in rows) / len(rows),
            "valid_attack_count": len(valid_failures),
            "source_answer_leakage_rate_on_valid_attacks": leakage / len(valid_failures) if valid_failures else None,
        }
        coverage[attack_type] = len(valid_failures)
        failures[attack_type] = [compact_example(row, sources[row["task_id"]])
                                 for row in valid_failures[:examples_per_type]]
    total = sum(len(rows) for rows in groups.values())
    return {
        "current_suffix": suffix, "selected_candidate_id": selected_id,
        "evaluation": {"model": evaluation["model"], "directions": total,
                       "full_accuracy": sum(row["full"]["correct"] for row in results) / total,
                       "reuse_accuracy": sum(row["reuse"]["correct"] for row in results) / total},
        "by_type": by_type, "coverage": {"historical_attack_pairs_by_type": coverage},
        "failure_examples": failures,
        "selection_note": "Examples satisfy Full correct and selected Reuse+suffix wrong; they are diagnostic context, not labels for generation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="candidate JSONL used for evaluation")
    parser.add_argument("--candidate", default="direct_reuse", help="direct_reuse or one candidate_id")
    parser.add_argument("--examples-per-type", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    if (args.run_dir is None) != (args.step is None):
        parser.error("--run-dir and --step must be provided together")
    if args.examples_per_type < 0:
        parser.error("--examples-per-type must be non-negative")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    sources = {row["task_id"]: row for row in (json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines())}
    dashboard = build_dashboard(evaluation, sources, candidate_id=args.candidate,
                                examples_per_type=args.examples_per_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.run_dir:
        manifest = write_step(args.run_dir, args.step, "feedback_dashboard", {
            "evaluation": file_ref(args.evaluation), "source_candidates": file_ref(args.source),
            "selected_candidate_id": dashboard["selected_candidate_id"],
            "examples_per_type": args.examples_per_type, "dashboard": file_ref(args.output),
        })
        print(f"step_manifest: {manifest}")
    print(json.dumps({"candidate": dashboard["selected_candidate_id"], "directions": dashboard["evaluation"]["directions"],
                      "valid_attacks": sum(d["valid_attack_count"] for d in dashboard["by_type"].values()),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())