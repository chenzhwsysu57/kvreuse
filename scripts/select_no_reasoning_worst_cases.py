#!/usr/bin/env python3
"""Select worst no-reasoning Deal reuse cases for fast calibration probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/reasoning_ablation/no_reasoning/qwen3-1.7b/samples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/deal_no_reasoning_worst.jsonl"))
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for line in args.results.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("dataset") != "deal_or_no_deal":
            continue
        pairs = [("a_to_b", "b"), ("b_to_a", "a")]
        losses = []
        disagreements = []
        for condition, target in pairs:
            full = row["full"][target]
            reuse = row["reuse"][condition]
            losses.append(int(full.get("correct", False) and not reuse.get("correct", False)))
            disagreements.append(int(full.get("prediction") != reuse.get("prediction")))
        reuse_values = [row["reuse"][name] for name, _ in pairs]
        score = (
            sum(losses),
            sum(disagreements),
            sum(int(v.get("prediction") is None) for v in reuse_values),
            sum(float(v.get("first_token_kl_full_to_reuse", 0.0)) for v in reuse_values),
        )
        selected = dict(row.get("input", {}))
        selected.update({"task_id": row["task_id"], "dataset": row["dataset"]})
        selected["debug_selection"] = {
            "losses_full_correct_reuse_wrong": sum(losses),
            "prediction_disagreements": sum(disagreements),
            "a_to_b_prediction": row["reuse"]["a_to_b"].get("prediction"),
            "a_to_b_gold": row["reuse"]["a_to_b"].get("gold"),
            "b_to_a_prediction": row["reuse"]["b_to_a"].get("prediction"),
            "b_to_a_gold": row["reuse"]["b_to_a"].get("gold"),
            "a_to_b_output": row["reuse"]["a_to_b"].get("output_text", ""),
            "b_to_a_output": row["reuse"]["b_to_a"].get("output_text", ""),
        }
        candidates.append((score, selected))

    ranked = sorted(candidates, key=lambda x: (x[0], x[1]["task_id"]), reverse=True)
    chosen = [row for _, row in ranked[: args.count]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"selected {len(chosen)} worst cases from {len(candidates)} Deal records")
    for row in chosen:
        print(row["task_id"], row["debug_selection"]["losses_full_correct_reuse_wrong"],
              row["debug_selection"]["a_to_b_prediction"], row["debug_selection"]["b_to_a_prediction"])
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
