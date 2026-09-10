#!/usr/bin/env python3
"""Summarize full vs cross-reuse accuracy drops for boss-facing tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def acc(side: dict[str, Any] | None) -> float | None:
    if not side:
        return None
    return float(side.get("accuracy", 0.0)) * 100.0


def row_from_pair(full_path: Path, reuse_path: Path, label: str, model: str, mode: str) -> dict[str, Any]:
    full = load_summary(full_path)
    reuse = load_summary(reuse_path)
    full_a = acc(full.get("metrics", {}).get("full", {}).get("a"))
    full_b = acc(full.get("metrics", {}).get("full", {}).get("b"))
    reuse_a_to_b = acc(reuse.get("metrics", {}).get("reuse", {}).get("a_to_b"))
    reuse_b_to_a = acc(reuse.get("metrics", {}).get("reuse", {}).get("b_to_a"))
    return {
        "dataset": label,
        "model": model,
        "mode": mode,
        "full_a": full_a,
        "full_b": full_b,
        "reuse_a_to_b": reuse_a_to_b,
        "reuse_b_to_a": reuse_b_to_a,
        "delta_a_to_b_vs_full_b": None if full_b is None or reuse_a_to_b is None else reuse_a_to_b - full_b,
        "delta_b_to_a_vs_full_a": None if full_a is None or reuse_b_to_a is None else reuse_b_to_a - full_a,
    }


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/reuse_drop_summary.json"))
    args = parser.parse_args()

    specs = [
        ("ArgKP", "benchmark_argkp_deal_harmbench_301", "argkp", 110, True),
        ("Deal", "benchmark_argkp_deal_harmbench_301", "deal_or_no_deal", 110, True),
        ("JobInterview", "job_interview_110_no_reasoning", "job_interview", 110, False),
        ("FANToM-belief", "fantom_110_no_reasoning", "fantom", 110, False),
        ("Perspectrum", "perspectrum_110_no_reasoning", "perspectrum", 110, False),
        ("ExploreToM", "explore_tom_44_no_reasoning", "explore_tom", 44, False),
        ("CaSiNo", "casino_110_no_reasoning", "casino", 110, False),
    ]

    rows: list[dict[str, Any]] = []
    for label, root, dataset_key, _count, has_reasoning_combo in specs:
        for model in ("1.7b", "4b"):
            for mode in ("no-reasoning", "reasoning"):
                if mode == "reasoning" and not has_reasoning_combo and label in {"ArgKP", "Deal"}:
                    subroot = args.results_root / root / "reasoning"
                elif mode == "reasoning":
                    subroot = args.results_root / root.replace("_no_reasoning", "_reasoning")
                else:
                    subroot = args.results_root / root
                model_dir = f"qwen3-{model}"
                full_path = subroot / model / "full" / model_dir / "summary.json"
                reuse_path = subroot / model / "reuse" / model_dir / "summary.json"
                if mode == "reasoning" and label in {"ArgKP", "Deal"}:
                    full_path = subroot / "full" / model_dir / "summary.json"
                    reuse_path = subroot / "reuse" / model_dir / "summary.json"
                    if not full_path.is_file():
                        continue
                if not full_path.is_file() or not reuse_path.is_file():
                    continue
                summary = row_from_pair(full_path, reuse_path, label, model, mode)
                if label in {"ArgKP", "Deal"} and mode == "reasoning":
                    by_dataset = load_summary(full_path).get("by_dataset", {}).get(dataset_key, {})
                    reuse_by_dataset = load_summary(reuse_path).get("by_dataset", {}).get(dataset_key, {})
                    summary["full_a"] = acc(by_dataset.get("full", {}).get("a"))
                    summary["full_b"] = acc(by_dataset.get("full", {}).get("b"))
                    summary["reuse_a_to_b"] = acc(reuse_by_dataset.get("reuse", {}).get("a_to_b"))
                    summary["reuse_b_to_a"] = acc(reuse_by_dataset.get("reuse", {}).get("b_to_a"))
                    if summary["full_b"] is not None and summary["reuse_a_to_b"] is not None:
                        summary["delta_a_to_b_vs_full_b"] = summary["reuse_a_to_b"] - summary["full_b"]
                    if summary["full_a"] is not None and summary["reuse_b_to_a"] is not None:
                        summary["delta_b_to_a_vs_full_a"] = summary["reuse_b_to_a"] - summary["full_a"]
                rows.append(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("| Dataset | Model | Mode | Full A | Full B | Reuse A→B | Reuse B→A | Δ(A→B vs Full B) | Δ(B→A vs Full A) |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['dataset']} | {row['model']} | {row['mode']} | "
            f"{fmt(row['full_a'])} | {fmt(row['full_b'])} | "
            f"{fmt(row['reuse_a_to_b'])} | {fmt(row['reuse_b_to_a'])} | "
            f"{fmt(row['delta_a_to_b_vs_full_b'])} | {fmt(row['delta_b_to_a_vs_full_a'])} |"
        )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
