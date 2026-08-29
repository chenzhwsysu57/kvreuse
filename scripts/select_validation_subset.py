#!/usr/bin/env python3
"""Select a stable dataset-validity subset shared by every model run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def rank(seed: int, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def choose_pku_balanced(
    candidates: list[dict[str, Any]], per_dataset: int, seed: int
) -> list[dict[str, Any]]:
    """Balance displayed gold while greedily diversifying categories/sources."""
    quotas = {"A": (per_dataset + 1) // 2, "B": per_dataset // 2}
    chosen: list[dict[str, Any]] = []
    pools = {
        label: [record for record in candidates if record["gold_a"] == label]
        for label in quotas
    }
    source_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    label_order = ["A", "B"] * ((per_dataset + 1) // 2)
    for label in label_order:
        if quotas[label] == 0:
            continue
        if not pools[label]:
            raise ValueError(f"pku_safe_rlhf lacks enough gold_a={label} records")
        record = min(
            pools[label],
            key=lambda item: (
                category_counts.get(item["metadata"]["primary_harm_category"], 0),
                source_counts.get(item["metadata"]["source_model"], 0),
                rank(seed, item["task_id"]),
            ),
        )
        pools[label].remove(record)
        chosen.append(record)
        quotas[label] -= 1
        source = record["metadata"]["source_model"]
        category = record["metadata"]["primary_harm_category"]
        source_counts[source] = source_counts.get(source, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    return sorted(chosen, key=lambda item: rank(seed, item["task_id"]))


def choose_harmbench_diverse(
    candidates: list[dict[str, Any]], per_dataset: int, seed: int
) -> list[dict[str, Any]]:
    """Diversify official semantic categories and context-length bands."""
    category_counts: dict[str, int] = {}
    length_counts: dict[str, int] = {}
    chosen: list[dict[str, Any]] = []
    remaining = list(candidates)
    while len(chosen) < per_dataset:
        if not remaining:
            raise ValueError("harmbench_contextual lacks enough records")

        def selection_key(item: dict[str, Any]) -> tuple[int, int, str]:
            metadata = item["metadata"]
            category = metadata["semantic_category"]
            length = int(metadata["context_characters"])
            length_band = "short" if length <= 512 else "medium" if length <= 1536 else "long"
            return (
                category_counts.get(category, 0),
                length_counts.get(length_band, 0),
                rank(seed, item["task_id"]),
            )

        record = min(remaining, key=selection_key)
        remaining.remove(record)
        chosen.append(record)
        metadata = record["metadata"]
        category = metadata["semantic_category"]
        length = int(metadata["context_characters"])
        length_band = "short" if length <= 512 else "medium" if length <= 1536 else "long"
        category_counts[category] = category_counts.get(category, 0) + 1
        length_counts[length_band] = length_counts.get(length_band, 0) + 1
    return sorted(chosen, key=lambda item: rank(seed, item["task_id"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/all.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/second_step_10_each.jsonl"))
    parser.add_argument("--per-dataset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = {}
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            grouped.setdefault(record["dataset"], []).append(record)

    selected: list[dict[str, Any]] = []
    for dataset in sorted(grouped):
        candidates = sorted(grouped[dataset], key=lambda item: rank(args.seed, item["task_id"]))
        if len(candidates) < args.per_dataset:
            raise ValueError(f"{dataset} has only {len(candidates)} records")
        if dataset == "pku_safe_rlhf":
            chosen = choose_pku_balanced(candidates, args.per_dataset, args.seed)
        elif dataset == "harmbench_contextual":
            chosen = choose_harmbench_diverse(candidates, args.per_dataset, args.seed)
        else:
            chosen = candidates[: args.per_dataset]
        selected.extend(chosen)
        print(f"{dataset}: {len(chosen)} selected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {len(selected)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
