#!/usr/bin/env python3
"""Distribute Full, direct Reuse and suffix evaluations across GPUs.

Each job runs in its own process, so its model and CUDA cache are released on
exit. GPU slot counts are user-controlled; the scheduler does not overrule them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
sys.path.insert(0, str(ROOT))

from kv_semantic_attack.adaptive_defense import parse_candidates
from kv_semantic_attack.run_journal import file_ref, write_step


@dataclass(frozen=True)
class Job:
    name: str
    kind: str
    suffix_file: Path | None
    output: Path


def parse_gpu_slots(values: list[str]) -> dict[int, int]:
    slots = {}
    for value in values:
        try:
            gpu, count = value.split("=", 1)
            gpu_id, slot_count = int(gpu), int(count)
            if gpu_id < 0 or slot_count < 1 or gpu_id in slots:
                raise ValueError
            slots[gpu_id] = slot_count
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid GPU slot {value!r}; use unique non-negative GPU=POSITIVE_COUNT") from None
    if not slots:
        raise argparse.ArgumentTypeError("at least one GPU slot is required")
    return slots


def gpu_usage() -> dict[int, tuple[int, int]]:
    command = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot query GPUs with nvidia-smi: {exc}") from exc
    result = {}
    for line in output.splitlines():
        index, used, total = (int(value.strip()) for value in line.split(","))
        result[index] = (used, total)
    return result


def materialize(candidates_path: Path, output_dir: Path) -> list[tuple[str, Path]]:
    candidates = parse_candidates(json.loads(candidates_path.read_text(encoding="utf-8")))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for candidate in candidates:
        path = output_dir / f"{candidate.candidate_id}.txt"
        path.write_text(candidate.suffix + "\n", encoding="utf-8")
        paths.append((candidate.candidate_id, path))
    return paths


def launch(job: Job, gpu: int, args: argparse.Namespace, log_dir: Path) -> subprocess.Popen:
    command = [str(PYTHON), "-u", "kv_semantic_attack/eval_synthetic_batch.py",
               "--method", "full" if job.kind == "full" else "reuse",
               "--model", args.model, "--batch-size", str(args.batch_size),
               "--max-new-tokens", str(args.max_new_tokens), "--input", str(args.input),
               "--output", str(job.output)]
    if job.kind != "full":
        command.extend(("--reuse-engine", "scatter"))
    if job.suffix_file is not None:
        command.extend(("--suffix-file", str(job.suffix_file)))
    path = log_dir / f"{job.name}.log"
    stream = path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    process._kvreuse_stream = stream  # type: ignore[attr-defined]
    process._kvreuse_started = time.monotonic()  # type: ignore[attr-defined]
    print(f"[dispatch] GPU {gpu}: {job.name} → {path}", flush=True)
    return process


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-slots", nargs="+", required=True, metavar="GPU=SLOTS",
                        help="maximum concurrent jobs per GPU, e.g. 0=2 1=1 2=2")
    parser.add_argument("--max-used-gib", type=float, default=4.0,
                        help="launch only when external GPU use is at most this amount; scheduler-owned jobs are ignored")
    parser.add_argument("--model", choices=("0.6b", "1.7b", "4b", "8b"), default="1.7b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    if (args.run_dir is None) != (args.step is None):
        parser.error("--run-dir and --step must be provided together")
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_used_gib < 0 or args.poll_seconds <= 0:
        parser.error("invalid batch, memory or polling setting")
    slots = parse_gpu_slots(args.gpu_slots)
    visible = gpu_usage()
    missing = set(slots) - set(visible)
    if missing:
        parser.error(f"requested GPU(s) not found: {sorted(missing)}; visible: {sorted(visible)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    suffixes = materialize(args.candidates, args.output_dir / "suffixes")
    jobs = deque([Job("full", "full", None, args.output_dir / "full.json"),
                  Job("direct_reuse", "reuse", None, args.output_dir / "direct_reuse.json"),
                  *[Job(name, "suffix", path, args.output_dir / f"suffix_{name}.json") for name, path in suffixes]])
    logs = args.output_dir / "logs"
    logs.mkdir()
    active: dict[int, list[tuple[Job, subprocess.Popen]]] = {gpu: [] for gpu in slots}
    completed = []
    while jobs or any(active.values()):
        for gpu, running in active.items():
            retained = []
            for job, process in running:
                status = process.poll()
                if status is None:
                    retained.append((job, process))
                    continue
                process._kvreuse_stream.close()  # type: ignore[attr-defined]
                duration = time.monotonic() - process._kvreuse_started  # type: ignore[attr-defined]
                completed.append({"name": job.name, "kind": job.kind, "gpu": gpu, "returncode": status,
                                  "seconds": duration, "output": str(job.output), "log": str(logs / f"{job.name}.log")})
                print(f"[complete] GPU {gpu}: {job.name} rc={status} {duration:.1f}s", flush=True)
                if status != 0:
                    print("[dispatch] stopping because a job failed; see its log", flush=True)
                    return 1
            active[gpu] = retained
        usage = gpu_usage()
        for gpu, maximum in slots.items():
            used, _ = usage[gpu]
            # nvidia-smi cannot attribute memory to this scheduler's child
            # processes. Gate only an idle GPU; once this scheduler owns a
            # slot, fill remaining user-authorized slots without mistaking its
            # own model allocations for external contention.
            if not active[gpu] and used > args.max_used_gib * 1024:
                continue
            while jobs and len(active[gpu]) < maximum:
                job = jobs.popleft()
                active[gpu].append((job, launch(job, gpu, args, logs)))
        if jobs or any(active.values()):
            time.sleep(args.poll_seconds)
    manifest = {"input": file_ref(args.input), "candidates": file_ref(args.candidates),
                "gpu_slots": slots, "max_used_gib": args.max_used_gib, "model": args.model,
                "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens, "jobs": completed}
    manifest_path = args.output_dir / "dispatch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.run_dir:
        step_manifest = write_step(args.run_dir, args.step, "multi_gpu_defense_dispatch", {
            **manifest, "dispatch_manifest": file_ref(manifest_path),
        })
        print(f"step_manifest: {step_manifest}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())