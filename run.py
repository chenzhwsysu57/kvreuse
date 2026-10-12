#!/usr/bin/env python3
"""Unified, auditable entry point for KV-reuse benchmark methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from kvreuse_data.schema import normalize_record

ROOT = Path(__file__).resolve().parent
RELAY_BATCH_RUNNER = Path(os.environ.get(
    "RELAY_BATCH_RUNNER",
    str(ROOT / "scripts" / "run_relay_methods.py"),
))
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "unified"
DATASETS = {
    # Historical synthetic-suite comments below describe the original private
    # registry; the public registry contains only the shipped benchmarks.
    # Generated synthetic suite. ``all`` is the complete 16-type, 4,800-pair
    # suite plus ArgKP, Deal, and Job Interview (5,130 pairs total).
    # Individual task aliases select one 300-pair synthetic task family.
    # The 220 suffix denotes 110 paired records / 220 A/B directional outputs.
    "argkp": ("data/benchmark/benchmark_argkp_deal_220.jsonl", "argkp", 110),
    "deal": ("data/benchmark/benchmark_argkp_deal_220.jsonl", "deal_or_no_deal", 110),
    "job": ("data/benchmark/benchmark_job_interview_110.jsonl", "job_interview", 110),
    "all330": ("data/benchmark/benchmark_final240_real_ood_330_20261006.jsonl", None, 330),
    "all240": ("data/benchmark/generated240/full_240classes_30each.jsonl", None, 7200),
    "train180": ("data/benchmark/generated240/train_180classes_30each.jsonl", None, 5400),
    "test60": ("data/benchmark/generated240/test_60classes_30each.jsonl", None, 1800),
}
METHOD_ALIASES = {
    "full": "full-prefill", "full-prefill": "full-prefill",
    "reuse": "reuse", "clean-reuse": "clean-reuse", "clean_reuse": "clean-reuse",
    "ours-post": "ours-post", "ours_post": "ours-post",
    "general-restate": "general-restate", "general_restate": "general-restate",
    "repeat-prefix": "repeat-prefix", "repeat_prefix": "repeat-prefix",
    "universal-soft-suffix": "universal-soft-suffix",
    "soft-prompt": "universal-soft-suffix", "remedy-prompt": "universal-soft-suffix",
    "kvcomm": "kvcomm", "relaycaching": "relaycaching",
    "cacheblend": "cacheblend", "epic": "epic",
}
RELAY_METHODS = {"relaycaching", "cacheblend", "epic"}
BATCH_METHODS = {"full-prefill", "reuse", "clean-reuse", "ours-post", "general-restate", "repeat-prefix"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--dataset", choices=sorted(DATASETS))
    selection.add_argument("--input", type=Path, help="custom paired-record JSONL")
    parser.add_argument("--input-label", help="custom input label: letters, digits, underscores, hyphens")
    parser.add_argument("--method", required=True, choices=sorted(METHOD_ALIASES))
    parser.add_argument("--model", default="4b", choices=("0.6b", "1.7b", "4b", "8b"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reasoning", required=True, choices=("yes", "no"))
    parser.add_argument("--kv-mode", required=True, choices=("fulla", "fullb", "a2b", "b2a", "all"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int, help="cap paired source records after dataset selection")
    parser.add_argument("--verify-limit", type=int, default=0,
                        help="compare this many batched records with batch-size one")
    parser.add_argument("--progress-accuracy", action="store_true",
                        help="show cumulative accuracy during universal-soft-suffix evaluation")
    parser.add_argument("--baseline", type=Path, help="full samples.jsonl required by KVCOMM")
    parser.add_argument("--soft-prompt-checkpoint", type=Path)
    parser.add_argument("--kvreuse-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--relay-python", type=Path,
                        default=Path(os.environ.get("RELAY_PYTHON", ROOT / ".venv-relay/bin/python")))
    parser.add_argument("--torch-runtime", type=Path,
                        default=Path(os.environ["KVREUSE_TORCH_RUNTIME"])
                        if os.environ.get("KVREUSE_TORCH_RUNTIME") else None,
                        help="optional node-local directory containing torch/nvidia/triton packages")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    if args.input_label is not None and args.input is None:
        parser.error("--input-label requires --input")
    if args.input is not None:
        args.input = args.input.expanduser().resolve()
        if not args.input.is_file():
            parser.error(f"input does not exist: {args.input}")
        if args.input_label is None:
            args.input_label = re.sub(r"[^A-Za-z0-9_-]+", "-", args.input.stem).strip("-_")[:64] or "input"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.input_label):
            parser.error("--input-label must be 1-64 safe characters, starting with a letter or digit")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [normalize_record(json.loads(line)) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_input(
    args: argparse.Namespace, run_dir: Path
) -> tuple[Path, list[dict[str, Any]], list[Path]]:
    if args.input is not None:
        sources = [args.input]
        rows = load_jsonl(args.input)
        if not rows:
            raise ValueError(f"input contains no records: {args.input}")
    else:
        relative, dataset_name, expected = DATASETS[args.dataset]
        sources = [ROOT / relative]
        rows = load_jsonl(sources[0])
        if dataset_name is not None:
            rows = [row for row in rows if row.get("dataset") == dataset_name]
        if len(rows) != expected:
            raise ValueError(f"{args.dataset} expected {expected} records, found {len(rows)} in {sources}")
    if args.limit is not None:
        rows = rows[:args.limit]
    snapshot = run_dir / "input.jsonl"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return snapshot, rows, sources


def input_output_label(args: argparse.Namespace) -> str:
    if args.input is None:
        return args.dataset
    path_digest = hashlib.sha256(str(args.input.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"custom-{args.input_label}-{path_digest}"


def run_directory(args: argparse.Namespace, method: str) -> Path:
    root = args.output_root.expanduser().resolve()
    mode = "reasoning" if args.reasoning == "yes" else "no_reasoning"
    directory = root / input_output_label(args) / mode / method / f"qwen3-{args.model}" / args.kv_mode.replace("_", "-")
    if not directory.resolve().is_relative_to(root):
        raise ValueError(f"run directory escapes output root: {directory}")
    if args.input is not None and args.input.resolve().is_relative_to(directory.resolve()):
        raise ValueError("custom input must not be inside its output directory")
    return directory


def modes(method: str, kv_mode: str) -> tuple[list[str], list[str]]:
    requested = [kv_mode] if kv_mode != "all" else (
        ["fulla", "fullb"] if method == "full-prefill" else ["a2b", "b2a"]
    )
    if method == "full-prefill":
        invalid = set(requested) - {"fulla", "fullb"}
        sides = ["a" if mode == "fulla" else "b" for mode in requested if mode in {"fulla", "fullb"}]
        directions: list[str] = []
    else:
        invalid = set(requested) - {"a2b", "b2a"}
        sides = ["b" if mode == "a2b" else "a" for mode in requested if mode in {"a2b", "b2a"}]
        directions = ["a_to_b" if mode == "a2b" else "b_to_a" for mode in requested]
    if invalid:
        raise ValueError(f"{method} does not support kv-mode(s): {sorted(invalid)}")
    return sides, directions


def build_command(
    args: argparse.Namespace, method: str, input_path: Path, run_dir: Path,
    target_sides: list[str], directions: list[str],
) -> list[str]:
    max_tokens = args.max_new_tokens or (4096 if args.reasoning == "yes" else 128)
    if method in BATCH_METHODS:
        internal = {
            "full-prefill": "full", "reuse": "reuse",
            "clean-reuse": "clean_reuse", "ours-post": "ours_post",
            "general-restate": "general_restate", "repeat-prefix": "repeat_prefix",
        }[method]
        command = [
            str(args.kvreuse_python), "-u", str(ROOT / "scripts/batch_eval.py"),
            "--model", args.model, "--input", str(input_path), "--method", internal,
            "--limit", str(len(load_jsonl(input_path))), "--batch-size", str(args.batch_size),
            "--verify-limit", str(args.verify_limit), "--max-new-tokens", str(max_tokens),
            "--target-sides", *target_sides, "--output", str(run_dir / "summary.json"),
        ]
        if args.reasoning == "yes":
            command.append("--explicit-reasoning")
        return command
    if method in RELAY_METHODS:
        command = [
            str(args.relay_python), "-u", str(RELAY_BATCH_RUNNER),
            "--method", method, "--model", args.model, "--input", str(input_path),
            "--output-root", str(run_dir / "artifacts"), "--limit", str(len(load_jsonl(input_path))),
            "--max-new-tokens", str(max_tokens),
            "--directions", *directions,
        ]
        if args.reasoning == "yes":
            command.append("--explicit-reasoning")
        if args.overwrite:
            command.append("--overwrite")
        return command
    if method == "kvcomm":
        if args.batch_size != 1:
            raise ValueError("kvcomm currently supports only --batch-size 1")
        if not args.baseline or not args.baseline.is_file():
            raise ValueError("kvcomm requires --baseline pointing to full samples.jsonl")
        command = [
            str(args.kvreuse_python), "-u", str(ROOT / "scripts/run_kvcomm.py"),
            "--model", args.model, "--input", str(input_path), "--baseline", str(args.baseline),
            "--output-root", str(run_dir / "artifacts"), "--max-samples", str(len(load_jsonl(input_path))),
            "--max-new-tokens", str(max_tokens), "--directions", *directions,
        ]
        if args.reasoning == "yes":
            command.append("--explicit-reasoning")
        if args.overwrite:
            command.append("--overwrite")
        return command
    if method == "universal-soft-suffix":
        if not args.soft_prompt_checkpoint or not args.soft_prompt_checkpoint.is_file():
            raise ValueError("universal-soft-suffix requires --soft-prompt-checkpoint")
        cache = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache/modelscope"))
        model_path = cache / f"models/Qwen--Qwen3-{args.model.upper()}/snapshots/master"
        command = [
            str(args.kvreuse_python), "-u", str(ROOT / "latent_remedy/eval_universal_soft_suffix.py"),
            "--checkpoint", str(args.soft_prompt_checkpoint), "--input", str(input_path),
            "--model-path", str(model_path), "--output", str(run_dir / "artifacts"),
            "--max-new-tokens", str(max_tokens), "--batch-size", str(args.batch_size),
            "--methods", "soft", "--eval-mode", "reasoning" if args.reasoning == "yes" else "no-reasoning",
            "--target-sides", *target_sides,
        ]
        if args.progress_accuracy:
            command.append("--progress-accuracy")
        return command
    raise AssertionError(method)


def main() -> int:
    args = parse_args()
    method = METHOD_ALIASES[args.method]
    target_sides, directions = modes(method, args.kv_mode)
    run_dir = run_directory(args, method)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"run directory is not empty: {run_dir}; use --overwrite")
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path, rows, sources = select_input(args, run_dir)
    command = build_command(args, method, input_path, run_dir, target_sides, directions)
    config = {
        "schema_version": 1, "dataset": args.dataset,
        "custom_input": str(args.input) if args.input else None,
        "input_label": args.input_label, "output_label": input_output_label(args),
        "dataset_sources": [str(source) for source in sources],
        "dataset_source_sha256": {str(source): sha256(source) for source in sources},
        "input_snapshot": str(input_path),
        "input_snapshot_sha256": sha256(input_path), "records": len(rows),
        "selected_task_ids": [row["task_id"] for row in rows], "method": method,
        "model": args.model, "batch_size": args.batch_size, "reasoning": args.reasoning,
        "kv_mode": args.kv_mode, "target_sides": target_sides, "directions": directions,
        "max_new_tokens": args.max_new_tokens or (4096 if args.reasoning == "yes" else 128),
        "baseline": str(args.baseline.resolve()) if args.baseline else None,
        "soft_prompt_checkpoint": (
            str(args.soft_prompt_checkpoint.resolve()) if args.soft_prompt_checkpoint else None
        ),
        "soft_prompt_checkpoint_sha256": (
            sha256(args.soft_prompt_checkpoint) if args.soft_prompt_checkpoint else None
        ),
        "kvreuse_python": str(args.kvreuse_python), "relay_python": str(args.relay_python),
        "torch_runtime": str(args.torch_runtime) if args.torch_runtime else None,
        "command": command,
    }
    write_json(run_dir / "run_config.json", config)
    print("RUN", " ".join(command), flush=True)
    if args.dry_run:
        return 0
    environment = os.environ.copy()
    python_paths = []
    if args.torch_runtime:
        if not args.torch_runtime.is_dir():
            raise FileNotFoundError(f"torch runtime directory not found: {args.torch_runtime}")
        python_paths.append(str(args.torch_runtime))
    python_paths.append(str(ROOT / "third_party/RelayCaching"))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = ":".join(python_paths)
    log_path = run_dir / "run.log"
    log_lines: list[str] = []
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log_lines.append(line)
        return_code = process.wait()
    except KeyboardInterrupt:
        print("\n[interrupt] stopping child process group", flush=True)
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return_code = 130
    finally:
        if log_lines:
            with log_path.open("a", encoding="utf-8") as log:
                log.writelines(log_lines)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())