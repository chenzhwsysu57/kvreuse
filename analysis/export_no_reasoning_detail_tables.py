#!/usr/bin/env python3
"""Export one wide accuracy table per Qwen3 model size and reasoning mode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("1.7b", "4b", "8b")
METHODS = (
    "full",
    "reuse",
    "clean_reuse",
    "tail16_recompute",
    "tail16_post_recompute",
    "ours_post",
    "kvcomm",
    "relaycaching",
    "cacheblend",
    "epic",
)
DATASETS = {
    "argkp-110": ("benchmark_argkp_deal_harmbench_301", "argkp", 110),
    "deal_or_no_deal-110": ("benchmark_argkp_deal_harmbench_301", "deal_or_no_deal", 110),
    "harmbench_contextual-81": ("benchmark_argkp_deal_harmbench_301", "harmbench_contextual", 81),
    "helpsteer2-133": ("helpsteer2_pku_safe_rlhf_266", "helpsteer2", 133),
    "pku_safe_rlhf-133": ("helpsteer2_pku_safe_rlhf_266", "pku_safe_rlhf", 133),
}


def board_path(results_dir: Path, model: str, mode: str) -> Path:
    if mode == "reasoning":
        return results_dir / f"reasoning-qwen3-{model}.csv"
    current = results_dir / f"no-reasoning-qwen3-{model}.csv"
    if current.is_file():
        return current
    return results_dir / f"done-no-reasoning-qwen3-{model}.csv"


def completed_direction_keys(rows: list[dict[str, object]], method: str) -> set[tuple[str, str]]:
    """Return distinct attempted A/B directions in an artifact's native schema."""
    completed: set[tuple[str, str]] = set()
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            continue
        if method == "kvcomm":
            completed.update((task_id, direction) for direction in row.get("calibration", {}))
        elif method == "full" and row.get("full"):
            completed.update((task_id, direction) for direction in row["full"])
        elif row.get("reuse"):
            completed.update((task_id, direction) for direction in row["reuse"])
        elif row.get("source_side") and row.get("target_side"):
            completed.add((task_id, f"{row['source_side']}_to_{row['target_side']}"))
    return completed


def incomplete_suffix(
    results_dir: Path, model: str, method: str, board_dataset: str, mode: str,
    completion_audit: dict[tuple[str, str, str, str], dict[str, str]],
) -> str:
    """Return ``(finished/expected)`` only for a locally visible partial run."""
    if mode != "reasoning":
        return ""
    audit = completion_audit.get((board_dataset, f"qwen3-{model}", method, "on"))
    if audit is not None:
        if audit["artifact_state"] == "partial":
            return f"({audit['successful_directions']}/{audit['expected_directions']})"
        return ""
    benchmark, artifact_dataset, records = DATASETS[board_dataset]
    samples = (
        results_dir
        / benchmark
        / "reasoning"
        / method
        / f"qwen3-{model}"
        / "samples.jsonl"
    )
    if not samples.is_file():
        # Some completed values were imported from Lq-Sakura without copying
        # samples locally, so their completion count is intentionally omitted.
        return ""
    rows = [
        json.loads(line)
        for line in samples.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line).get("dataset") == artifact_dataset
    ]
    finished = len(completed_direction_keys(rows, method))
    expected = records * 2
    return f"({finished}/{expected})" if 0 < finished < expected else ""


def export_table(
    board: Path, destination: Path, results_dir: Path, model: str, mode: str,
    completion_audit: dict[tuple[str, str, str, str], dict[str, str]],
) -> None:
    values: dict[tuple[str, str], str] = {}
    with board.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values[(row["dataset"], row["method"])] = row["status"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", *METHODS))
        writer.writeheader()
        for dataset in DATASETS:
            row_values = {}
            for method in METHODS:
                value = values.get((dataset, method), "")
                if value.endswith("%"):
                    value += incomplete_suffix(results_dir, model, method, dataset, mode, completion_audit)
                row_values[method] = value
            writer.writerow(
                {
                    "dataset": dataset,
                    **row_values,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--mode",
        choices=("no-reasoning", "reasoning"),
        default="no-reasoning",
        help="Which experiment mode to export (default: no-reasoning).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--completion-audit",
        type=Path,
        help="Optional merged audit CSV; its remote-aware counts override local artifact counts.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or Path(
        "analysis/outputs/"
        + ("reasoning_detail_tables" if args.mode == "reasoning" else "no_reasoning_detail_tables")
    )
    completion_audit: dict[tuple[str, str, str, str], dict[str, str]] = {}
    if args.completion_audit:
        with args.completion_audit.open(newline="") as handle:
            for row in csv.DictReader(handle):
                completion_audit[(row["dataset"], row["model"], row["method"], row["reasoning"])] = row

    for model in MODELS:
        source = board_path(args.results_dir, model, args.mode)
        if not source.is_file():
            raise FileNotFoundError(source)
        filename_mode = args.mode.replace("-", "_")
        output = output_dir / f"qwen3-{model}_{filename_mode}_detail.csv"
        export_table(source, output, args.results_dir, model, args.mode, completion_audit)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
