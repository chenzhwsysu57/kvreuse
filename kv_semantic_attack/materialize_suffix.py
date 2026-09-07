#!/usr/bin/env python3
"""Write validated suffix candidates from a defender JSON file to text files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kv_semantic_attack.adaptive_defense import parse_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    candidates = parse_candidates(json.loads(args.candidates.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        (args.output_dir / f"{candidate.candidate_id}.txt").write_text(candidate.suffix + "\n", encoding="utf-8")
    print("\n".join(str(args.output_dir / f"{candidate.candidate_id}.txt") for candidate in candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())