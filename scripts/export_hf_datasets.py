#!/usr/bin/env python3
"""Materialize Hugging Face datasets into pinned JSONL files under data/raw/."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


CRAIGSLIST_BARGAINS_REV = "main"
EXPLORE_TOM_REV = "main"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def export_craigslist_bargains(output_dir: Path) -> list[Path]:
    from datasets import load_dataset

    dataset = load_dataset("stanfordnlp/craigslist_bargains", trust_remote_code=True)
    paths: list[Path] = []
    for split, rows in dataset.items():
        path = output_dir / "craigslist_bargains" / f"{split}.jsonl"
        write_jsonl(path, [dict(row) for row in rows])
        paths.append(path)
    return paths


def export_explore_tom(output_dir: Path) -> list[Path]:
    from datasets import load_dataset

    rows = load_dataset("facebook/ExploreToM", split="train")
    path = output_dir / "explore_tom" / "train.jsonl"
    write_jsonl(path, [dict(row) for row in rows])
    return [path]


EXPORTERS = {
    "craigslist_bargains": export_craigslist_bargains,
    "explore_tom": export_explore_tom,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(EXPORTERS),
        default=sorted(EXPORTERS),
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    )
    args = parser.parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    try:
        import datasets  # noqa: F401
    except ImportError as error:
        print("export_hf_datasets.py requires the `datasets` package", file=sys.stderr)
        raise SystemExit(1) from error

    for name in args.datasets:
        paths = EXPORTERS[name](args.output_dir)
        for path in paths:
            print(f"exported {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
