#!/usr/bin/env python3
"""Create a per-dataset accuracy table from benchmark method artifacts."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results/benchmark_argkp_deal_harmbench_301/no_reasoning"
METHODS = (
    "full", "reuse", "clean_reuse", "tail16_recompute", "tail16_post_recompute", "ours_post",
    "ours_precaution", "ours_repeat_txt", "ours_repeat_kv", "kvcomm", "relaycaching", "cacheblend", "epic",
)
DATASETS = ("argkp", "deal_or_no_deal", "harmbench_contextual")
LABELS = {
    "full": "Full", "reuse": "Reuse", "clean_reuse": "Clean", "tail16_recompute": "Tail-16",
    "tail16_post_recompute": "Tail-16 + post", "kvcomm": "KVCOMM",
    "ours_post": "Ours-post", "ours_precaution": "Ours-precaution", "relaycaching": "Relay",
    "ours_repeat_txt": "Ours-repeat-text", "ours_repeat_kv": "Ours-repeat-KV",
    "cacheblend": "CacheBlend", "epic": "EPIC",
}


def correctness(row: dict, method: str) -> list[tuple[str, bool]]:
    """Normalise direct-runner paired rows and Relay single-direction rows."""
    dataset = row["dataset"]
    if method == "kvcomm":
        return [(dataset, bool(value["correct"])) for value in row["calibration"].values()]
    if method == "full" and row.get("full"):
        return [(dataset, bool(value["correct"])) for value in row["full"].values()]
    if row.get("reuse"):
        return [(dataset, bool(value["correct"])) for value in row["reuse"].values()]
    return [(dataset, bool(row["correct"]))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default="qwen3-1.7b")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS,
                        help="methods to include; defaults to every known method")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS,
                        help="datasets to include in the table and overall row")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    methods = tuple(args.methods)
    datasets = tuple(args.datasets)
    aggregate: dict[str, dict[str, list[bool]]] = {}
    for method in methods:
        path = args.root / method / args.model_dir / "samples.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        values: dict[str, list[bool]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                for dataset, correct in correctness(json.loads(line), method):
                    values[dataset].append(correct)
        aggregate[method] = values

    header = ["dataset (outputs)", *[LABELS[method] for method in methods]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for dataset in (*datasets, "overall"):
        entries = []
        for method in methods:
            values = (sum((aggregate[method][name] for name in datasets), [])
                      if dataset == "overall" else aggregate[method][dataset])
            entries.append(f"{100 * sum(values) / len(values):.2f}%")
        n = len(sum((aggregate[methods[0]][name] for name in datasets), [])) if dataset == "overall" else len(aggregate[methods[0]][dataset])
        lines.append("| " + " | ".join([f"{dataset} ({n})", *entries]) + " |")
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
