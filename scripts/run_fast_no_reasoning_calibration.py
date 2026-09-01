#!/usr/bin/env python3
"""Fast no-reasoning calibration on the worst Deal reuse cases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="1.7b", choices=("0.6b", "1.7b", "4b", "8b"))
    parser.add_argument("--input", type=Path, default=Path("data/validation/deal_no_reasoning_worst.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("results/deal_fast_no_reasoning"))
    parser.add_argument("--counts", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    runner = Path(__file__).with_name("run_direct_reuse.py")
    for position in ("prefix_block", "end"):
        for count in args.counts:
            output = args.output_root / f"{position}_{count}nl"
            command = [
                sys.executable, "-u", str(runner), "--model", args.model,
                "--input", str(args.input), "--output-root", str(output),
                "--max-new-tokens", str(args.max_new_tokens), "--boxed-output",
                "--newline-position", position, "--newline-count", str(count),
                "--no-plots", "--overwrite",
            ]
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
