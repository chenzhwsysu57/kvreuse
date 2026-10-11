#!/usr/bin/env python3
"""Evaluate a trained latent remedy prompt on every default benchmark task.

Defaults evaluate every task in the public 240-class generated suite plus ArgKP, Deal or
No Deal, and Job Interview.  Training task IDs recorded in the checkpoint's
split manifest are always excluded, so results never include a training pair.
Each task name remains a separate row in the evaluator summary.

Examples
--------
python test-latent-token.py \
  --checkpoint results/latent_example/best_latent_tokens.pt \
  --eval-mode no-reasoning

python test-latent-token.py \
  --checkpoint results/latent_example/best_latent_tokens.pt \
  --eval-mode reasoning --max-new-tokens 4096
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = (
    ROOT / "data/benchmark/generated240/full_240classes_30each.jsonl",
    ROOT / "data/benchmark/benchmark_argkp_deal_220.jsonl",
    ROOT / "data/benchmark/benchmark_job_interview_110.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append",
                        help="override default benchmark inputs; repeat for multiple files")
    parser.add_argument("--output", type=Path,
                        help="default: <checkpoint-dir>/test_<eval-mode>")
    parser.add_argument("--eval-mode", choices=("no-reasoning", "reasoning"), default="no-reasoning")
    parser.add_argument("--max-new-tokens", type=int,
                        help="default: 32/no-reasoning or 4096/reasoning")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="paired records per batch; A/B together form twice as many directions")
    parser.add_argument("--methods", nargs="+", choices=("full", "direct", "soft"), default=("soft",),
                        help="default soft only; Full and Direct are evaluated separately by run.py")
    parser.add_argument("--model-path", type=Path,
                        help="explicit model snapshot; defaults to MODELSCOPE_CACHE based on checkpoint model")
    parser.add_argument("--torch-runtime", type=Path,
                        default=Path(os.environ["KVREUSE_TORCH_RUNTIME"])
                        if os.environ.get("KVREUSE_TORCH_RUNTIME") else None,
                        help="optional torch/nvidia/triton packages; set explicitly or via KVREUSE_TORCH_RUNTIME")
    parser.add_argument("--allow-training-examples", action="store_true",
                        help="include training task IDs; disabled by default to prevent leakage")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens is not None and args.max_new_tokens < 1 or args.batch_size < 1:
        parser.error("--max-new-tokens and --batch-size must be positive")
    return args


def checkpoint_config(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), str):
        raise ValueError(f"{path}: missing checkpoint training config/model")
    return config


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    config = checkpoint_config(checkpoint)
    inputs = tuple(path.resolve() for path in (args.input or DEFAULT_INPUTS))
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing evaluation inputs: {', '.join(missing)}")
    output = (args.output or checkpoint.parent / f"test_{args.eval_mode}").resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    max_tokens = args.max_new_tokens or (4096 if args.eval_mode == "reasoning" else 32)
    if args.model_path:
        model_path = args.model_path.resolve()
    else:
        cache = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache/modelscope"))
        model_path = cache / f"models/Qwen--Qwen3-{config['model'].upper()}" / "snapshots/master"
    manifest = checkpoint.parent / "split_manifest.json"
    if not args.allow_training_examples and not manifest.is_file():
        raise FileNotFoundError(
            f"missing {manifest}; use --allow-training-examples only if intentional"
        )
    command = [
        sys.executable, "-u", str(ROOT / "kv_semantic_attack/eval_universal_soft_suffix.py"),
        "--checkpoint", str(checkpoint), "--model-path", str(model_path),
        "--output", str(output), "--eval-mode", args.eval_mode,
        "--max-new-tokens", str(max_tokens), "--batch-size", str(args.batch_size),
        "--methods", *args.methods,
    ]
    for path in inputs:
        command.extend(("--input", str(path)))
    if not args.allow_training_examples:
        command.extend(("--exclude-task-id-file", str(manifest)))
    plan = {
        "checkpoint": str(checkpoint),
        "model": config["model"],
        "inputs": [str(path) for path in inputs],
        "eval_mode": args.eval_mode,
        "max_new_tokens": max_tokens,
        "batch_size_pairs": args.batch_size,
        "methods": list(args.methods),
        "excluded_training_task_ids": not args.allow_training_examples,
        "command": command,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    environment = os.environ.copy()
    if args.torch_runtime is not None:
        if not args.torch_runtime.is_dir():
            raise FileNotFoundError(f"torch runtime directory not found: {args.torch_runtime}")
        python_paths = [str(args.torch_runtime), str(ROOT / "third_party/RelayCaching")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = ":".join(python_paths)
        print(f"[setup] using torch runtime: {args.torch_runtime}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
