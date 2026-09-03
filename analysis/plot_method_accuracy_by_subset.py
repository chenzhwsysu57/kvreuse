#!/usr/bin/env python3
"""Plot per-subset method accuracy from the six progress CSV boards.

Each panel is square and covers one Qwen3 size / reasoning-mode combination.
Within a panel, subsets are ordered by their mean completed-method accuracy.
Missing ``PD`` and ``R-*`` cells are deliberately omitted rather than imputed.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "full", "reuse", "clean_reuse", "tail16_recompute", "tail16_post_recompute",
    "ours_post", "kvcomm", "relaycaching", "cacheblend", "epic",
)
METHOD_LABELS = {
    "full": "Full",
    "reuse": "Reuse",
    "clean_reuse": "Clean reuse",
    "tail16_recompute": "Tail-16",
    "tail16_post_recompute": "Tail-16 + post",
    "ours_post": "Ours-post",
    "kvcomm": "KVCOMM",
    "relaycaching": "RelayCaching",
    "cacheblend": "CacheBlend",
    "epic": "EPIC",
}
DATASET_LABELS = {
    "argkp-110": "ArgKP\n(110)",
    "deal_or_no_deal-110": "Deal\n(110)",
    "harmbench_contextual-81": "HarmBench\n(81)",
    "helpsteer2-133": "HelpSteer2\n(133)",
    "pku_safe_rlhf-133": "PKU-SafeRLHF\n(133)",
}
MODELS = ("1.7b", "4b", "8b")
MODES = (("reasoning", "reasoning"), ("no_reasoning", "no-reasoning"))
MARKERS = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "*")
COLORS = plt.get_cmap("tab10").colors


def parse_accuracy(status: str) -> float | None:
    """Return a completed percentage, leaving PD/R cells missing."""
    if not status.endswith("%"):
        return None
    try:
        return float(status[:-1])
    except ValueError:
        return None


def load_board(path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            accuracy = parse_accuracy(row["status"])
            if accuracy is not None and row["method"] in METHODS:
                values[row["dataset"]][row["method"]] = accuracy
    return values


def plot_board(values: dict[str, dict[str, float]], *, model: str, mode_label: str, output: Path) -> None:
    # Dataset "overall" means the mean across available completed methods in
    # this exact board; it is used only to choose a stable left-to-right order.
    ordered_datasets = sorted(
        values,
        key=lambda dataset: (-np.mean(list(values[dataset].values())), dataset),
    )
    displayed = [accuracy for dataset in ordered_datasets for accuracy in values[dataset].values()]
    if not displayed:
        raise ValueError(f"no completed values for Qwen3-{model}, {mode_label}")
    y_min, y_max = min(displayed), max(displayed)
    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5

    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    positions = np.arange(len(ordered_datasets), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    for index, method in enumerate(METHODS):
        method_x = []
        method_y = []
        for position, dataset in enumerate(ordered_datasets):
            accuracy = values[dataset].get(method)
            if accuracy is not None:
                method_x.append(positions[position] + offsets[index])
                method_y.append(accuracy)
        if method_x:
            axis.scatter(
                method_x, method_y, label=METHOD_LABELS[method], marker=MARKERS[index],
                color=COLORS[index], edgecolors="black", linewidths=0.35, s=62, zorder=3,
            )

    axis.set_title(f"Qwen3-{model} · {mode_label}", weight="bold")
    axis.set_xlabel("Dataset subset (ordered by mean completed-method accuracy, high → low)")
    axis.set_ylabel("Accuracy (%)")
    axis.set_xticks(positions, [DATASET_LABELS.get(dataset, dataset) for dataset in ordered_datasets])
    axis.set_ylim(y_min, y_max)
    axis.set_xlim(-0.55, len(ordered_datasets) - 0.45)
    axis.set_box_aspect(1)
    axis.grid(axis="y", color="0.88", linewidth=0.8, zorder=0)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analysis" / "outputs" / "method_accuracy_by_subset",
    )
    args = parser.parse_args()

    for model in MODELS:
        for file_prefix, mode_label in MODES:
            if file_prefix == "no_reasoning":
                board_path = args.results_dir / f"no-reasoning-qwen3-{model}.csv"
                if not board_path.is_file():
                    # Older completed boards were named ``done-no-reasoning``.
                    board_path = args.results_dir / f"done-no-reasoning-qwen3-{model}.csv"
            else:
                board_path = args.results_dir / f"{file_prefix}-qwen3-{model}.csv"
            if not board_path.is_file():
                raise FileNotFoundError(board_path)
            output = args.output_dir / f"qwen3-{model}_{file_prefix}_accuracy.png"
            plot_board(load_board(board_path), model=model, mode_label=mode_label, output=output)
            print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
