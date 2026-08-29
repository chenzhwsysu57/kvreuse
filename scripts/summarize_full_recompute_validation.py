#!/usr/bin/env python3
"""Aggregate completed no-reuse dataset-validity runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/full_recompute_validation"))
    args = parser.parse_args()
    model_dirs = sorted(path for path in args.root.glob("qwen3-*") if path.is_dir())
    table = []
    audit = {"models": {}, "total_outputs": 0}
    for model_dir in model_dirs:
        config = json.loads((model_dir / "run_config.json").read_text(encoding="utf-8"))
        rows = load_jsonl(model_dir / "predictions.jsonl")
        expected_outputs = int(config.get("expected_outputs", 60))
        if len(rows) != expected_outputs:
            raise ValueError(f"{model_dir}: expected {expected_outputs} outputs, got {len(rows)}")
        if config.get("enable_thinking") is not False:
            raise ValueError(f"{model_dir}: thinking is not disabled")
        task_sides = {(row["task_id"], row["side"]) for row in rows}
        if len(task_sides) != expected_outputs:
            raise ValueError(f"{model_dir}: duplicate task/side rows")
        if any(row["enable_thinking"] for row in rows):
            raise ValueError(f"{model_dir}: a row enabled thinking")
        if any(row["hit_max_new_tokens"] for row in rows):
            raise ValueError(f"{model_dir}: a row hit max_new_tokens")
        if any(row["messages"][0] != {"role": "system", "content": "You're a helpful assistant."} for row in rows):
            raise ValueError(f"{model_dir}: incorrect system message")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["dataset"]].append(row)
        audit["models"][config["model_id"]] = {"outputs": len(rows), "input_sha256": config["input_sha256"]}
        audit["total_outputs"] += len(rows)
        for dataset, values in sorted(grouped.items()):
            pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for value in values:
                pairs[value["task_id"]][value["side"]] = value
            if len(pairs) != 10 or any(set(pair) != {"a", "b"} for pair in pairs.values()):
                raise ValueError(f"{model_dir}/{dataset}: expected ten complete A/B pairs")
            correct = sum(bool(value["correct"]) for value in values)
            parsed = sum(value["prediction"] is not None for value in values)
            both_correct = sum(pair["a"]["correct"] and pair["b"]["correct"] for pair in pairs.values())
            changed = sum(pair["a"]["prediction"] != pair["b"]["prediction"] for pair in pairs.values())
            table.append({
                "model": config["model_id"],
                "dataset": dataset,
                "outputs": len(values),
                "accuracy": correct / len(values),
                "parse_rate": parsed / len(values),
                "both_sides_correct_rate": both_correct / len(pairs),
                "prediction_change_rate": changed / len(pairs),
                "truncations": sum(bool(value["hit_max_new_tokens"]) for value in values),
            })

    input_hashes = {value["input_sha256"] for value in audit["models"].values()}
    audit["same_validation_subset"] = len(input_hashes) == 1
    if not audit["same_validation_subset"]:
        raise ValueError("models did not use the same validation subset")
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "summary_all.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (args.root / "summary_all.json").write_text(
        json.dumps({"audit": audit, "groups": table}, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Full-recompute dataset validity summary",
        "",
        "Thinking mode is disabled. Each model used the same 10 examples per dataset and evaluated both prefix sides.",
        "",
        "| Model | Dataset | Accuracy | Both sides correct | Prediction changed | Parse rate | Truncations |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['accuracy']:.0%} | "
            f"{row['both_sides_correct_rate']:.0%} | {row['prediction_change_rate']:.0%} | "
            f"{row['parse_rate']:.0%} | {row['truncations']} |"
        )
    (args.root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nAudited {audit['total_outputs']} outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
