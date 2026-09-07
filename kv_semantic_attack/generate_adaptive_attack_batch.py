#!/usr/bin/env python3
"""Program-generate one attacker-selected distribution without a model/GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kv_semantic_attack.adaptive_attack import AttackDistribution, allocate_counts, generate_attack_batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True, help="attacker JSON distribution")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    proposal = AttackDistribution.from_dict(json.loads(args.proposal.read_text(encoding="utf-8")))
    records = generate_attack_batch(proposal, seed=args.seed, round_index=args.round)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(json.dumps({"pairs": len(records), "counts": allocate_counts(proposal),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())