#!/usr/bin/env python3
"""Create readable method-comparison heatmaps from progress CSV boards.

For every Qwen3 size and reasoning setting, the script writes two square
figures: absolute accuracy and the paired change relative to Full.  Missing
``PD``/``R-*`` cells remain visibly missing and are never imputed.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("1.7b", "4b", "8b")
MODES = (("reasoning", "reasoning"), ("no_reasoning", "no-reasoning"))
METHODS = (
    "full", "reuse", "clean_reuse", "tail16_recompute", "tail16_post_recompute",
    "ours_post", "kvcomm", "relaycaching", "cacheblend", "epic",
)
METHOD_LABELS = {
    "full": "Full", "reuse": "Reuse", "clean_reuse": "Clean reuse",
    "tail16_recompute": "Tail-16", "tail16_post_recompute": "Tail-16 + post",
    "ours_post": "Ours-post", "kvcomm": "KVCOMM", "relaycaching": "RelayCaching",
    "cacheblend": "CacheBlend", "epic": "EPIC",
}
DATASET_LABELS = {
    "argkp-110": "ArgKP\n(110)", "deal_or_no_deal-110": "Deal\n(110)",
    "harmbench_contextual-81": "HarmBench\n(81)", "helpsteer2-133": "HelpSteer2\n(133)",
    "pku_safe_rlhf-133": "PKU-SafeRLHF\n(133)",
}


def parse_accuracy(status: str) -> float | None:
    if not status.endswith("%"):
        return None
    try:
        return float(status[:-1])
    except ValueError:
        return None


def board_path(results_dir: Path, model: str, file_prefix: str) -> Path:
    if file_prefix == "no_reasoning":
        current = results_dir / f"no-reasoning-qwen3-{model}.csv"
        return current if current.is_file() else results_dir / f"done-no-reasoning-qwen3-{model}.csv"
    return results_dir / f"reasoning-qwen3-{model}.csv"


def load_board(path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            accuracy = parse_accuracy(row["status"])
            if row["method"] in METHODS and accuracy is not None:
                values[row["dataset"]][row["method"]] = accuracy
    return values


def ordered_axes(values: dict[str, dict[str, float]]) -> tuple[list[str], list[str]]:
    datasets = sorted(
        DATASET_LABELS,
        key=lambda dataset: (-np.mean(list(values.get(dataset, {}).values())) if values.get(dataset) else float("inf"), dataset),
    )
    method_means = {}
    for method in METHODS:
        completed = [per_dataset[method] for per_dataset in values.values() if method in per_dataset]
        method_means[method] = np.mean(completed) if completed else float("nan")
    methods = sorted(METHODS, key=lambda method: (-method_means[method] if not np.isnan(method_means[method]) else float("inf"), method))
    return methods, datasets


def accuracy_matrix(values: dict[str, dict[str, float]], methods: list[str], datasets: list[str]) -> np.ma.MaskedArray:
    matrix = np.full((len(methods), len(datasets)), np.nan)
    for row, method in enumerate(methods):
        for column, dataset in enumerate(datasets):
            if method in values.get(dataset, {}):
                matrix[row, column] = values[dataset][method]
    return np.ma.masked_invalid(matrix)


def delta_matrix(accuracy: np.ma.MaskedArray, methods: list[str]) -> np.ma.MaskedArray:
    full_row = methods.index("full")
    full = accuracy[full_row]
    delta = np.ma.masked_all(accuracy.shape)
    for row in range(accuracy.shape[0]):
        available = ~(np.ma.getmaskarray(accuracy[row]) | np.ma.getmaskarray(full))
        delta[row, available] = accuracy[row, available] - full[available]
    return delta


def annotate(axis: plt.Axes, matrix: np.ma.MaskedArray, *, formatter) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if not np.ma.getmaskarray(matrix)[row, column]:
                axis.text(column, row, formatter(float(matrix[row, column])), ha="center", va="center", fontsize=8)
            else:
                axis.text(column, row, "—", ha="center", va="center", color="0.48", fontsize=10)


def draw_heatmap(
    matrix: np.ma.MaskedArray, *, methods: list[str], datasets: list[str], title: str,
    colorbar_label: str, cmap: str, output: Path, norm=None, formatter=str,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad("#e7e7e7")
    image = axis.imshow(matrix, cmap=palette, norm=norm, aspect="auto")
    # Ten methods by five subsets: square cells make method differences easier
    # to scan while the overall canvas remains square.
    axis.set_box_aspect(len(methods) / len(datasets))
    axis.set_title(title, weight="bold", pad=12)
    axis.set_xticks(range(len(datasets)), [DATASET_LABELS[dataset] for dataset in datasets])
    axis.set_yticks(range(len(methods)), [METHOD_LABELS[method] for method in methods])
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=8)
    axis.tick_params(axis="y", length=0)
    axis.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.4)
    axis.tick_params(which="minor", bottom=False, left=False)
    annotate(axis, matrix, formatter=formatter)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.76, pad=0.025)
    colorbar.set_label(colorbar_label)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analysis" / "outputs" / "method_accuracy_heatmaps",
    )
    args = parser.parse_args()

    boards = []
    all_deltas = []
    for model in MODELS:
        for prefix, mode_label in MODES:
            path = board_path(args.results_dir, model, prefix)
            if not path.is_file():
                raise FileNotFoundError(path)
            values = load_board(path)
            methods, datasets = ordered_axes(values)
            accuracy = accuracy_matrix(values, methods, datasets)
            delta = delta_matrix(accuracy, methods)
            boards.append((model, prefix, mode_label, methods, datasets, accuracy, delta))
            all_deltas.extend(delta.compressed())

    # A shared symmetric scale makes red/blue intensity comparable across all
    # six relative-to-Full figures.
    delta_limit = max(1.0, float(np.max(np.abs(all_deltas))))
    delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)
    for model, prefix, mode_label, methods, datasets, accuracy, delta in boards:
        stem = f"qwen3-{model}_{prefix}"
        draw_heatmap(
            accuracy, methods=methods, datasets=datasets,
            title=f"Qwen3-{model} · {mode_label} · accuracy",
            colorbar_label="Accuracy (%)", cmap="YlGnBu",
            norm=plt.Normalize(vmin=0, vmax=100), formatter=lambda value: f"{value:.1f}",
            output=args.output_dir / f"{stem}_accuracy_heatmap.png",
        )
        draw_heatmap(
            delta, methods=methods, datasets=datasets,
            title=f"Qwen3-{model} · {mode_label} · change vs Full",
            colorbar_label="Accuracy change vs Full (pp)", cmap="RdYlGn",
            norm=delta_norm, formatter=lambda value: f"{value:+.1f}",
            output=args.output_dir / f"{stem}_vs_full_heatmap.png",
        )
        print(f"wrote {args.output_dir / f'{stem}_accuracy_heatmap.png'}")
        print(f"wrote {args.output_dir / f'{stem}_vs_full_heatmap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
