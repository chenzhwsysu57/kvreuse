#!/usr/bin/env python3
"""Continuously schedule a 2-model × 2-mode × 7-method format-switch matrix.

A GPU becomes eligible after being below 50% utilisation for five consecutive
seconds and using at most 12 GiB. The dispatcher may then add a task even when
it already owns a low-utilisation child there, but only if observed usage plus
the incoming task's conservative model budget stays within 23 GiB.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/benchmark/benchmark_synthetic_format_switch_option_200/all.jsonl"
KVREUSE_PYTHON = ROOT / ".venv_kvreuse/bin/python"
RELAY_PYTHON = ROOT / ".venv_relaycaching/bin/python"
METHODS = ("full", "reuse", "ours_post", "kvcomm", "relaycaching", "epic", "cacheblend")
MODELS = ("1.7b", "4b")
MODES = ("no_reasoning", "reasoning")
# Observed loading budgets rounded upward, including transient KV buffers.
BUDGET_MIB = {"1.7b": 8_500, "4b": 15_500}
TOTAL_PAIRS = 200
MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class Task:
    model: str
    mode: str
    method: str

    @property
    def reasoning(self) -> bool:
        return self.mode == "reasoning"

    @property
    def max_new_tokens(self) -> int:
        return 512 if self.reasoning else 128


def gpu_state() -> dict[int, tuple[int, int]]:
    command = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True)
    return {int(i): (int(mem), int(util)) for i, mem, util in (line.split(",") for line in output.splitlines())}


def result_path(root: Path, task: Task) -> Path:
    base = root / task.model / task.mode / task.method
    if task.method == "full":
        return base / f"qwen3-{task.model}/summary.json"
    if task.method in {"reuse", "ours_post"}:
        return base / "evaluation.json"
    if task.method == "kvcomm":
        return base / f"qwen3-{task.model}/summary.json"
    relay_base = base / "reasoning" if task.reasoning else base
    return relay_base / task.method / f"qwen3-{task.model}/summary.json"


def full_samples(root: Path, task: Task) -> Path:
    return root / task.model / task.mode / "full" / f"qwen3-{task.model}" / "samples.jsonl"


def command(root: Path, task: Task) -> list[str]:
    base = root / task.model / task.mode / task.method
    if task.method == "full":
        args = [str(KVREUSE_PYTHON), "-u", str(ROOT / "scripts/run_direct_reuse.py"),
                "--input", str(INPUT), "--output-root", str(base), "--model", task.model,
                "--method", "full", "--boxed-output", "--max-new-tokens", str(task.max_new_tokens), "--no-plots"]
        if task.reasoning:
            args += ["--explicit-reasoning"]
        return args
    if task.method in {"reuse", "ours_post"}:
        args = [str(KVREUSE_PYTHON), "-u", str(ROOT / "kv_semantic_attack/eval_synthetic_batch.py"),
                "--input", str(INPUT), "--output", str(base / "evaluation.json"), "--model", task.model,
                "--method", "reuse", "--batch-size", "16",
                "--max-new-tokens", str(task.max_new_tokens)]
        args += ["--reuse-engine", "scatter"]
        if task.method == "ours_post":
            args += ["--suffix-file", str(ROOT / "scripts/format_option_post.txt")]
        if task.reasoning:
            args += ["--explicit-reasoning"]
        return args
    if task.method == "kvcomm":
        args = [str(KVREUSE_PYTHON), "-u", str(ROOT / "scripts/run_kvcomm.py"), "--model", task.model,
                "--input", str(INPUT), "--baseline", str(full_samples(root, task)),
                "--output-root", str(base), "--max-new-tokens", str(task.max_new_tokens)]
        if task.reasoning:
            args += ["--explicit-reasoning"]
        return args
    args = [str(RELAY_PYTHON), "-u", str(ROOT / "scripts/run_relay_methods.py"), "--method", task.method,
            "--model", task.model, "--input", str(INPUT), "--output-root", str(base),
            "--max-new-tokens", str(task.max_new_tokens)]
    if task.reasoning:
        args += ["--explicit-reasoning"]
    return args


def ready(root: Path, task: Task) -> bool:
    if task.method != "kvcomm":
        return True
    samples = full_samples(root, task)
    if not samples.is_file():
        return False
    with samples.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip()) == TOTAL_PAIRS


def completed(root: Path, task: Task) -> bool:
    """Return true only for a final artifact, never a direct-runner checkpoint."""
    if task.method == "full":
        samples = full_samples(root, task)
        return samples.is_file() and sum(1 for line in samples.open(encoding="utf-8") if line.strip()) == TOTAL_PAIRS
    path = result_path(root, task)
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return report.get("complete", True) is True


def reset_incomplete_output(root: Path, task: Task) -> None:
    """Remove only non-resumable partial reports before a retry."""
    if task.method not in {"reuse", "ours_post"}:
        return
    path = result_path(root, task)
    if path.exists() and not completed(root, task):
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--max-used-gib", type=float, default=12.0)
    parser.add_argument("--util-threshold", type=int, default=50)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--low-util-seconds", type=int, default=5)
    args = parser.parse_args()
    if not INPUT.is_file() or not KVREUSE_PYTHON.is_file() or not RELAY_PYTHON.is_file():
        raise FileNotFoundError("missing input or required Python environment")
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = [Task(model, mode, method) for model in MODELS for mode in MODES for method in METHODS]
    manifest_path = args.output_root / "task_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({
        "input": str(INPUT), "pairs": 200, "directions": 400, "boxed_answer": True,
        "models": MODELS, "modes": MODES, "methods": METHODS, "tasks": [asdict(t) for t in tasks],
        "policy": {"gpus": args.gpus, "low_util_seconds": args.low_util_seconds,
                   "util_threshold_percent": args.util_threshold, "max_used_gib": args.max_used_gib,
                   "admission_limit_mib": 23 * 1024, "budget_mib": BUDGET_MIB},
        }, indent=2) + "\n")
    env = os.environ | {"MODELSCOPE_CACHE": str(ROOT / ".modelscope"), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                        "PYTHONPATH": str(ROOT / "third_party/RelayCaching")}
    pending = [task for task in tasks if not completed(args.output_root, task)]
    active: dict[int, list[tuple[Task, subprocess.Popen, object]]] = {gpu: [] for gpu in args.gpus}
    low_for = {gpu: 0 for gpu in args.gpus}
    events = []
    attempts = {task: 0 for task in tasks}
    while pending or any(active.values()):
        state = gpu_state()
        for gpu, children in active.items():
            retained = []
            for task, proc, stream in children:
                status = proc.poll()
                if status is None:
                    retained.append((task, proc, stream))
                    continue
                stream.close()
                events.append({"event": "complete", "gpu": gpu, "task": asdict(task), "returncode": status})
                if status:
                    attempts[task] += 1
                    if attempts[task] >= MAX_ATTEMPTS:
                        (args.output_root / "dispatch.json").write_text(json.dumps(events, indent=2) + "\n")
                        raise RuntimeError(f"task failed {MAX_ATTEMPTS} times: {task}")
                    reset_incomplete_output(args.output_root, task)
                    pending.insert(0, task)
                    events.append({"event": "retry", "task": asdict(task), "attempt": attempts[task]})
            active[gpu] = retained
        for gpu in args.gpus:
            used, util = state[gpu]
            low_for[gpu] = low_for[gpu] + 1 if util < args.util_threshold and used <= args.max_used_gib * 1024 else 0
            if low_for[gpu] < args.low_util_seconds:
                continue
            # A batch-16 4B reuse process can share with one 1.7B worker when
            # live memory remains under the 23 GiB admission limit, but two 4B
            # workers on the same 24 GiB card cause OOM.
            has_4b = any(task.model == "4b" for task, _, _ in active[gpu])
            candidate = next((task for task in pending
                              if ready(args.output_root, task)
                              and not (task.model == "4b" and has_4b)
                              and not (task.model == "1.7b" and has_4b)
                              and used + BUDGET_MIB[task.model] <= 23 * 1024), None)
            if candidate is None:
                continue
            pending.remove(candidate)
            log = args.output_root / "logs" / f"{candidate.model}_{candidate.mode}_{candidate.method}.log"
            log.parent.mkdir(exist_ok=True)
            stream = log.open("a", encoding="utf-8")
            proc = subprocess.Popen(command(args.output_root, candidate), cwd=ROOT,
                                    env=env | {"CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=stream, stderr=subprocess.STDOUT)
            active[gpu].append((candidate, proc, stream))
            events.append({"event": "launch", "gpu": gpu, "task": asdict(candidate), "pid": proc.pid})
            low_for[gpu] = 0
            print(f"[launch] GPU {gpu}: {candidate}", flush=True)
        (args.output_root / "dispatch.json").write_text(json.dumps(events, indent=2) + "\n")
        if pending or any(active.values()):
            time.sleep(args.sample_seconds)
    print("All 28 tasks completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
