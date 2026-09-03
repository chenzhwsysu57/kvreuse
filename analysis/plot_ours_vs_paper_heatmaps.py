#!/usr/bin/env python3
"""Plot direct, per-subset accuracy margins of our variants over paper baselines.

Full is intentionally excluded: every cell is ``our variant - paper method``
on the same model, reasoning mode, and dataset subset.  Missing results stay
grey, so no incomplete result is interpreted as a loss.
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
OURS = ("ours_post", "tail16_recompute", "tail16_post_recompute")
PAPERS = ("kvcomm", "relaycaching", "cacheblend", "epic")
OURS_LABELS = {
    "ours_post": "Ours-post",
    "tail16_recompute": "Tail-16",
    "tail16_post_recompute": "Tail-16 + post",
}
PAPER_LABELS = {
    "kvcomm": "KVCOMM",
    "relaycaching": "RelayCaching",
    "cacheblend": "CacheBlend",
    "epic": "EPIC",
}
DATASET_LABELS = {
    "argkp-110": "ArgKP", "deal_or_no_deal-110": "Deal",
    "harmbench_contextual-81": "HarmBench", "helpsteer2-133": "HelpSteer2",
    "pku_safe_rlhf-133": "PKU-SafeRLHF",
}


def completed_accuracy(status: str) -> float | None:
    if not status.endswith("%"):
        return None
    try:
        return float(status[:-1])
    except ValueError:
        return None


def find_board(results_dir: Path, model: str, prefix: str) -> Path:
    if prefix == "reasoning":
        return results_dir / f"reasoning-qwen3-{model}.csv"
    current = results_dir / f"no-reasoning-qwen3-{model}.csv"
    return current if current.is_file() else results_dir / f"done-no-reasoning-qwen3-{model}.csv"


def load_values(path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            accuracy = completed_accuracy(row["status"])
            if accuracy is not None and row["method"] in {*OURS, *PAPERS}:
                values[row["dataset"]][row["method"]] = accuracy
    return values


def paper_order(values: dict[str, dict[str, float]]) -> list[str]:
    """Put subsets with stronger paper-baseline performance first."""
    def score(dataset: str) -> tuple[float, str]:
        completed = [values[dataset][method] for method in PAPERS if method in values[dataset]]
        return (-float(np.mean(completed)) if completed else float("inf"), dataset)
    return sorted(DATASET_LABELS, key=score)


def margin_matrix(values: dict[str, dict[str, float]], datasets: list[str]) -> tuple[np.ma.MaskedArray, list[str]]:
    rows = [(ours, dataset) for ours in OURS for dataset in datasets]
    matrix = np.full((len(rows), len(PAPERS)), np.nan)
    for row, (ours, dataset) in enumerate(rows):
        ours_accuracy = values.get(dataset, {}).get(ours)
        if ours_accuracy is None:
            continue
        for column, paper in enumerate(PAPERS):
            paper_accuracy = values.get(dataset, {}).get(paper)
            if paper_accuracy is not None:
                matrix[row, column] = ours_accuracy - paper_accuracy
    return np.ma.masked_invalid(matrix), [f"{OURS_LABELS[ours]}  |  {DATASET_LABELS[dataset]}" for ours, dataset in rows]


def plot_panel(matrix: np.ma.MaskedArray, row_labels: list[str], *, title: str, norm: TwoSlopeNorm, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 10))
    figure.subplots_adjust(left=0.34, right=0.86, top=0.93, bottom=0.05)
    palette = plt.get_cmap("RdYlGn").copy()
    palette.set_bad("#e7e7e7")
    image = axis.imshow(matrix, cmap=palette, norm=norm, aspect="auto")
    axis.set_title(title, weight="bold", pad=12)
    axis.set_xticks(range(len(PAPERS)), [PAPER_LABELS[method] for method in PAPERS])
    axis.set_yticks(range(len(row_labels)), row_labels, fontsize=8.5)
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=8)
    axis.tick_params(axis="y", length=0)
    axis.set_xticks(np.arange(-0.5, len(PAPERS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.3)
    axis.tick_params(which="minor", bottom=False, left=False)
    # Strong separators make the three variants readable as distinct groups.
    for boundary in (4.5, 9.5):
        axis.axhline(boundary, color="black", linewidth=1.4)
    mask = np.ma.getmaskarray(matrix)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            text = "—" if mask[row, column] else f"{float(matrix[row, column]):+.1f}"
            axis.text(column, row, text, ha="center", va="center", fontsize=8.5,
                      color="0.48" if mask[row, column] else "black")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.052, pad=0.04)
    colorbar.set_label("Our method − paper method accuracy (pp)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analysis" / "outputs" / "ours_vs_paper_methods",
    )
    args = parser.parse_args()

    panels = []
    all_margins = []
    for model in MODELS:
        for prefix, mode_label in MODES:
            path = find_board(args.results_dir, model, prefix)
            if not path.is_file():
                raise FileNotFoundError(path)
            values = load_values(path)
            datasets = paper_order(values)
            matrix, labels = margin_matrix(values, datasets)
            panels.append((model, prefix, mode_label, matrix, labels))
            all_margins.extend(matrix.compressed())

    limit = max(1.0, float(np.max(np.abs(all_margins))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for model, prefix, mode_label, matrix, labels in panels:
        output = args.output_dir / f"qwen3-{model}_{prefix}_ours_vs_papers.png"
        plot_panel(
            matrix, labels, title=f"Qwen3-{model} · {mode_label} · ours vs paper methods",
            norm=norm, output=output,
        )
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
