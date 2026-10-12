#!/usr/bin/env python3
"""Explicit, idempotent release-data migration; preserves all prompts and golds."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kvreuse_data.schema import validate_record

FIELDS = ("dataset", "prefix_a", "prefix_b", "shared_block", "question", "gold_a", "gold_b")


def compact(data: Path) -> dict[str, int]:
    paths = sorted(data.rglob("*.jsonl"))
    records = {p: [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
                   if line.strip()] for p in paths}
    families = sorted({r["dataset"] for r in records[data / "benchmark/generated240/full_240classes_30each.jsonl"]})
    if len(families) != 240:
        raise ValueError("expected 240 full-corpus families")
    anonymous = all(re.fullmatch(r"family_\d{3}", name) for name in families)
    mapping = {name: name if anonymous else f"family_{index:03d}"
               for index, name in enumerate(families, 1)}
    outputs = {}
    before_bytes = 0
    for path, rows in records.items():
        clean = []
        for row in rows:
            result = {key: row[key] for key in FIELDS}
            result["dataset"] = mapping.get(result["dataset"], result["dataset"])
            if row.get("output_protocol") not in (None, "option_letter"):
                result["output_protocol"] = row["output_protocol"]
            validate_record(result)
            clean.append(result)
        outputs[path] = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in clean)
        before_bytes += path.stat().st_size
    # Validate all records before replacing any file; never create a manifest.
    for path, text in outputs.items():
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return {"files": len(paths), "records": sum(map(len, records.values())),
            "bytes_before": before_bytes, "bytes_after": sum(p.stat().st_size for p in paths)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(json.dumps(compact(args.data_root), indent=2))