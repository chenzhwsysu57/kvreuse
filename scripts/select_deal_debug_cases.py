"""Select suspicious Deal-or-No-Deal reuse cases for KV calibration debugging."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/reasoning_ablation/reasoning/qwen3-1.7b/samples.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/validation/deal_debug_cases.jsonl"))
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for row in load_rows(args.results):
        if row.get("dataset") != "deal_or_no_deal":
            continue
        for condition, result in row.get("reuse", {}).items():
            text = str(result.get("output_text", ""))
            lower = text.lower()
            mentions_a = bool(re.search(r"\bagent\s+a\b", lower))
            mentions_b = bool(re.search(r"\bagent\s+b\b", lower))
            both_agents = mentions_a and mentions_b
            wrong = not bool(result.get("correct", False))
            unparsed = result.get("prediction") is None
            # Prefer semantic agent confusion, then incorrect/unparsed outputs,
            # and finally longer traces where the failure is easier to inspect.
            score = (
                int(both_agents),
                int(unparsed),
                int(wrong),
                int(result.get("hit_max_new_tokens", False)),
                min(len(text.split()), 1000),
            )
            # Convert the result row back to the runner's input schema.  The
            # inference JSONL stores prefix/block/question under `input`,
            # whereas run_direct_reuse expects those fields at top level.
            selected = dict(row.get("input", {}))
            selected["task_id"] = row["task_id"]
            selected["dataset"] = row["dataset"]
            if "metric" in row:
                selected["metric"] = row["metric"]
            if "metadata" in row:
                selected["metadata"] = row["metadata"]
            selected["debug_selection"] = {
                "condition": condition,
                "prediction": result.get("prediction"),
                "gold": result.get("gold"),
                "correct": result.get("correct"),
                "mentions_agent_a": mentions_a,
                "mentions_agent_b": mentions_b,
                "output_words": len(text.split()),
                "output_text": text,
            }
            candidates.append((score, selected))

    # Keep at most one direction per task so each selected record remains a
    # valid runner input while the debug metadata identifies the bad direction.
    best_by_task: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    for score, row in candidates:
        task_id = row["task_id"]
        if task_id not in best_by_task or score > best_by_task[task_id][0]:
            best_by_task[task_id] = (score, row)
    by_condition: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {
        "a_to_b": [], "b_to_a": []
    }
    for item in best_by_task.values():
        by_condition[item[1]["debug_selection"]["condition"]].append(item)
    for values in by_condition.values():
        values.sort(key=lambda item: (item[0], item[1]["task_id"]), reverse=True)
    # Prefer a balanced direction set so one direction cannot monopolize the
    # debugging corpus merely because its traces happen to be longer.
    quota_a = (args.count + 1) // 2
    quota_b = args.count // 2
    chosen_items = by_condition["a_to_b"][:quota_a] + by_condition["b_to_a"][:quota_b]
    chosen = [row for _, row in chosen_items]
    chosen.sort(key=lambda row: row["task_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"selected {len(chosen)} cases from {len(candidates)} suspicious directions")
    for row in chosen:
        info = row["debug_selection"]
        print(
            f"{row['task_id']} {info['condition']} pred={info['prediction']} "
            f"gold={info['gold']} words={info['output_words']} "
            f"both_agents={info['mentions_agent_a'] and info['mentions_agent_b']}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
