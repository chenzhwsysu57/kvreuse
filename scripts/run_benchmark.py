#!/usr/bin/env python3
"""Run one KV-reuse benchmark method on benchmark_250.

Examples
--------
  # Dense full baseline, no visible reasoning.
  python scripts/run_benchmark.py --method full --reasoning no --model 1.7b

  # RoPE-corrected direct cross-reuse only; dense outputs are not regenerated.
  python scripts/run_benchmark.py --method reuse --reasoning yes --model 4b

  # One official RelayCaching baseline in its dedicated environment.
  python scripts/run_benchmark.py --method cacheblend --reasoning no --model 1.7b

``reuse`` is the project's direct A->B/B->A transplant with inverse-source and
target-position RoPE correction. ``clean_reuse`` applies the same transplant
but creates block KV without any source prefix. ``relaycaching``, ``cacheblend``, and
``epic`` invoke their official RelayCaching code path through its separate
environment. ``ours_post`` and ``ours_precaution`` invoke the standalone bridge
ablation. Every invocation runs exactly one method unless ``--method all`` is
requested explicitly.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/benchmark/benchmark_250.jsonl"
METHODS = (
    "full", "reuse", "clean_reuse", "tail16_recompute", "tail16_post_recompute",
    "ours_post", "ours_precaution", "ours_repeat_txt", "ours_repeat_kv",
    "kvcomm",
    "relaycaching", "cacheblend", "epic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=(*METHODS, "all"))
    parser.add_argument("--reasoning", required=True, choices=("no", "yes"))
    parser.add_argument("--model", required=True, choices=("0.6b", "1.7b", "4b", "8b"))
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/benchmark_250")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test cap on input records before directions.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kvreuse-python", type=Path, default=Path("/home/czw/miniconda3/envs/kvreuse/bin/python"))
    parser.add_argument("--relay-python", type=Path, default=Path("/home/czw/miniconda3/envs/relaycaching/bin/python"))
    return parser.parse_args()


def command_for(args: argparse.Namespace, method: str) -> tuple[list[str], dict[str, str]]:
    explicit_reasoning = args.reasoning == "yes"
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (1024 if explicit_reasoning else 128)
    mode_dir = "reasoning" if explicit_reasoning else "no_reasoning"
    common = ["--model", args.model, "--input", str(args.input), "--max-new-tokens", str(max_new_tokens)]
    if args.limit is not None:
        common += ["--max-samples", str(args.limit)]
    if args.overwrite:
        common.append("--overwrite")

    if method in {
        "full", "reuse", "clean_reuse", "tail16_recompute", "tail16_post_recompute",
        "ours_repeat_txt", "ours_repeat_kv",
    }:
        command = [
            str(args.kvreuse_python), "-u", str(ROOT / "scripts/run_direct_reuse.py"),
            "--method", method,
            "--output-root", str(args.output_root / mode_dir / method),
            "--no-plots", "--boxed-output", *common,
        ]
        if explicit_reasoning:
            command.append("--explicit-reasoning")
        return command, os.environ.copy()

    if method in {"ours_post", "ours_precaution"}:
        bridge = "post_task_restatement" if method == "ours_post" else "pre_block_precaution"
        command = [
            str(args.kvreuse_python), "-u", str(ROOT / "scripts/run_ours_bridge_reuse.py"),
            "--bridge", bridge,
            "--method", "all",
            "--output-root", str(args.output_root / mode_dir / method),
            "--no-plots", "--boxed-output", *common,
        ]
        if explicit_reasoning:
            command.append("--explicit-reasoning")
        return command, os.environ.copy()

    if method == "kvcomm":
        # KVCOMM performs its prefix-anchor calibration against a completed
        # direct full-prefill run.  Keeping that artifact read-only lets this
        # method be scheduled independently while still using exactly the
        # same records and prompt configuration as Full.
        baseline = args.output_root / mode_dir / "full" / f"qwen3-{args.model}" / "samples.jsonl"
        command = [
            str(args.kvreuse_python), "-u", str(ROOT / "scripts/run_kvcomm.py"),
            "--model", args.model,
            "--input", str(args.input),
            "--baseline", str(baseline),
            "--output-root", str(args.output_root / mode_dir / method),
            "--max-new-tokens", str(max_new_tokens),
        ]
        if args.limit is not None:
            command += ["--max-samples", str(args.limit)]
        if args.overwrite:
            command.append("--overwrite")
        if explicit_reasoning:
            command.append("--explicit-reasoning")
        return command, os.environ.copy()

    # RelayCaching's own runner calls its argument ``--limit`` rather than
    # run_direct_reuse's ``--max-samples``.
    relay_common = ["--model", args.model, "--input", str(args.input), "--max-new-tokens", str(max_new_tokens)]
    if args.limit is not None:
        relay_common += ["--limit", str(args.limit)]
    if args.overwrite:
        relay_common.append("--overwrite")
    command = [
        str(args.relay_python), "-u", str(ROOT / "scripts/run_relay_methods.py"),
        "--method", method,
        # run_relay_methods.py itself inserts ``reasoning/`` for visible-CoT
        # runs, whereas no-reasoning paths are written directly underneath its
        # output root.
        "--output-root", str(args.output_root if explicit_reasoning else args.output_root / mode_dir),
        *relay_common,
    ]
    if explicit_reasoning:
        command.append("--explicit-reasoning")
    environment = os.environ.copy()
    relay_root = str(ROOT / "third_party/RelayCaching")
    environment["PYTHONPATH"] = relay_root + (":" + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    return command, environment


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"benchmark input not found: {args.input}; run scripts/prepare_benchmark.py")
    methods = METHODS if args.method == "all" else (args.method,)
    for method in methods:
        command, environment = command_for(args, method)
        print("RUN", " ".join(command), flush=True)
        if not args.dry_run:
            if method == "kvcomm":
                baseline = args.output_root / ("reasoning" if args.reasoning == "yes" else "no_reasoning") / "full" / f"qwen3-{args.model}" / "samples.jsonl"
                if not baseline.is_file():
                    raise FileNotFoundError(
                        f"KVCOMM requires a completed Full baseline at {baseline}. "
                        "Run --method full first, or use --method all."
                    )
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
