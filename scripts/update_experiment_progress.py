#!/usr/bin/env python3
"""Build six CSV progress boards for the five maintained benchmark datasets.

Completed cells contain their per-dataset accuracy; ``R`` means an artifact
exists but has not completed all requested records, and ``PD`` means no
artifact has been produced yet.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen3-1.7b", "qwen3-4b", "qwen3-8b")
METHODS = (
    "full", "reuse", "clean_reuse", "tail16_recompute", "tail16_post_recompute",
    "ours_post", "kvcomm", "relaycaching", "cacheblend", "epic",
)
BENCHMARKS = {
    "benchmark_argkp_deal_harmbench_301": {
        "argkp-110": ("argkp", 110),
        "deal_or_no_deal-110": ("deal_or_no_deal", 110),
        "harmbench_contextual-81": ("harmbench_contextual", 81),
    },
    "helpsteer2_pku_safe_rlhf_266": {
        "helpsteer2-133": ("helpsteer2", 133),
        "pku_safe_rlhf-133": ("pku_safe_rlhf", 133),
    },
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def correctness(row: dict[str, Any], method: str) -> list[bool]:
    """Normalize the direct, bridge, KVCOMM, and RelayCaching artifacts."""
    if method == "kvcomm":
        return [bool(value["correct"]) for value in row.get("calibration", {}).values()]
    if method == "full" and row.get("full"):
        return [bool(value["correct"]) for value in row["full"].values()]
    if row.get("reuse"):
        return [bool(value["correct"]) for value in row["reuse"].values()]
    return [bool(row["correct"])] if "correct" in row else []


def status_for(
    results_root: Path, benchmark: str, dataset: str, expected_records: int,
    model: str, method: str, reasoning_dir: str, process_commands: list[str],
    remote_running: dict[tuple[str, str, str, str], str],
    remote_results: dict[tuple[str, str, str, str, str], str],
) -> str:
    remote_result = remote_results.get((benchmark, reasoning_dir, model, method, dataset))
    if remote_result is not None:
        return remote_result
    artifact_dir = results_root / benchmark / reasoning_dir / method / model
    samples_path = artifact_dir / "samples.jsonl"
    rows = load_rows(samples_path)
    dataset_rows = [row for row in rows if row.get("dataset") == dataset]
    completed_ids = {row.get("task_id") for row in dataset_rows if row.get("task_id")}
    if len(completed_ids) < expected_records:
        remote_status = remote_running.get((benchmark, reasoning_dir, model, method))
        if remote_status is not None:
            return remote_status
        model_size = model.removeprefix("qwen3-")
        process_markers = (
            f"{benchmark}/{reasoning_dir}",
            f"--method {method}",
            f"--model {model_size}",
        )
        return "R" if any(all(marker in command for marker in process_markers) for command in process_commands) else "PD"
    values = [correct for row in dataset_rows for correct in correctness(row, method)]
    if not values:
        return "R"
    return f"{100 * sum(values) / len(values):.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--remote-running", action="append", default=[], metavar="BENCHMARK:MODE:MODEL:METHOD:STATUS",
        help="mark an incomplete remote run, e.g. benchmark_argkp_deal_harmbench_301:reasoning:qwen3-8b:reuse:R-Lq-Sakura",
    )
    parser.add_argument(
        "--remote-result", action="append", default=[], metavar="BENCHMARK:MODE:MODEL:METHOD:DATASET:ACCURACY",
        help="insert a remotely summarized completed result without copying its artifacts",
    )
    args = parser.parse_args()

    remote_running: dict[tuple[str, str, str, str], str] = {}
    for value in args.remote_running:
        parts = value.split(":", 4)
        if len(parts) != 5 or not all(parts):
            raise ValueError("--remote-running must be BENCHMARK:MODE:MODEL:METHOD:STATUS")
        benchmark, mode, model, method, status = parts
        remote_running[(benchmark, mode, model, method)] = status
    remote_results: dict[tuple[str, str, str, str, str], str] = {}
    for value in args.remote_result:
        parts = value.split(":", 5)
        if len(parts) != 6 or not all(parts):
            raise ValueError("--remote-result must be BENCHMARK:MODE:MODEL:METHOD:DATASET:ACCURACY")
        benchmark, mode, model, method, dataset, accuracy = parts
        remote_results[(benchmark, mode, model, method, dataset)] = accuracy

    process_listing = subprocess.run(
        ["ps", "-eo", "args="], check=True, text=True, capture_output=True
    ).stdout
    process_commands = process_listing.splitlines()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for reasoning, reasoning_dir in (("off", "no_reasoning"), ("on", "reasoning")):
            output = args.output_dir / f"{reasoning_dir.replace('_', '-')}-{model}.csv"
            with output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("dataset", "model", "method", "reasoning", "status"))
                for benchmark, datasets in BENCHMARKS.items():
                    for dataset_name, (dataset, expected_records) in datasets.items():
                        for method in METHODS:
                            writer.writerow((
                                dataset_name, model, method, reasoning,
                                status_for(
                                    args.results_root, benchmark, dataset, expected_records,
                                    model, method, reasoning_dir, process_commands, remote_running, remote_results,
                                ),
                            ))
            print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
