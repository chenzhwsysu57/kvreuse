#!/usr/bin/env python3
"""Compare source-prefix dependence between correct and wrong Deal reuse cases.

For every Deal A->B/B->A direction in a direct RoPE-reuse result, this script
re-prefills the *source* prompt with eager attention.  It groups directions by
whether cross-reuse answered correctly, then compares shared-block attention to
the source prefix versus preceding block tokens.  Block lengths are resampled
to fixed relative-position bins before aggregation, so long blocks do not
dominate the group mean.

Run on a GPU node after the no-reasoning full/reuse benchmark is available:

  /home/czw/miniconda3/envs/kvreuse/bin/python \
    analysis/compare_deal_reuse_attention.py --model 1.7b
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ANALYSIS = ROOT / "analysis"
for path in (SCRIPTS, ANALYSIS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from plot_deal_prefix_attention import attention_masses  # noqa: E402
from run_direct_reuse import MODEL_IDS, build_prompt_parts, load_jsonl, validate_model_files  # noqa: E402


METRICS = (
    "prefix_attention_mass",
    "prior_block_attention_mass",
    "prefix_share_of_history",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--reasoning", choices=("no", "yes"), default="no")
    parser.add_argument("--benchmark-input", type=Path, default=ROOT / "data/benchmark/benchmark_250.jsonl")
    parser.add_argument("--reuse-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=64, help="Relative shared-block position bins used for group comparison.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max-per-group", type=int, default=0, help="0 keeps every correct and wrong direction.")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def resample_relative_positions(matrix: np.ndarray, bins: int) -> np.ndarray:
    """Mean-pool a [layers, variable tokens] matrix into [layers, bins]."""
    layers, tokens = matrix.shape
    output = np.full((layers, bins), np.nan, dtype=np.float64)
    for bin_index in range(bins):
        start = int(np.floor(bin_index * tokens / bins))
        end = int(np.floor((bin_index + 1) * tokens / bins))
        end = max(start + 1, end)
        output[:, bin_index] = np.nanmean(matrix[:, start:end], axis=1)
    return output


def select_directions(records: list[dict[str, Any]], reuse_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records_by_id = {row["task_id"]: row for row in records}
    selected: list[dict[str, Any]] = []
    for row in reuse_rows:
        if row.get("dataset") != "deal_or_no_deal":
            continue
        record = records_by_id.get(row["task_id"])
        if record is None:
            raise KeyError(f"{row['task_id']} absent from benchmark input")
        for direction, source, target in (("a_to_b", "a", "b"), ("b_to_a", "b", "a")):
            result = row.get("reuse", {}).get(direction)
            if result is None:
                raise ValueError(f"{row['task_id']} missing {direction}; expected direct reuse output")
            selected.append({
                "task_id": row["task_id"], "direction": direction,
                "source_side": source, "target_side": target,
                "reuse_correct": bool(result["correct"]),
                "gold": result["gold"], "reuse_prediction": result["prediction"],
                "reuse_output": result["output_text"], "record": record,
            })
    return selected


def bootstrap_difference(correct: np.ndarray, wrong: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    """95% CI of wrong-minus-correct for one scalar per direction."""
    if samples <= 0:
        return float("nan"), float("nan")
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        wrong_mean = rng.choice(wrong, size=len(wrong), replace=True).mean()
        correct_mean = rng.choice(correct, size=len(correct), replace=True).mean()
        differences[index] = wrong_mean - correct_mean
    return tuple(float(value) for value in np.quantile(differences, (0.025, 0.975)))


def plot_groups(output_path: Path, group_matrices: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "prefix_attention_mass": "Attention mass on source prefix",
        "prior_block_attention_mass": "Attention mass on preceding block",
        "prefix_share_of_history": "Prefix share of historical attention",
    }
    figure, axes = plt.subplots(len(METRICS), 3, figsize=(18, 12), constrained_layout=True, squeeze=False)
    for row_index, metric in enumerate(METRICS):
        correct = group_matrices["correct"][metric].mean(axis=0)
        wrong = group_matrices["wrong"][metric].mean(axis=0)
        difference = wrong - correct
        for column, (matrix, title, cmap, limits) in enumerate((
            (correct, "Reuse correct", "viridis", (0.0, 1.0)),
            (wrong, "Reuse wrong", "viridis", (0.0, 1.0)),
            (difference, "Wrong − correct", "coolwarm", (-0.5, 0.5)),
        )):
            axis = axes[row_index, column]
            image = axis.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, vmin=limits[0], vmax=limits[1])
            axis.set_title(f"{title}: {labels[metric]}")
            axis.set_xlabel("Relative shared-block position bin")
            axis.set_ylabel("Layer index")
            figure.colorbar(image, ax=axis, shrink=0.78)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.bins <= 0:
        raise ValueError("--bins must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Attention extraction requires a CUDA GPU node")
    mode = "reasoning" if args.reasoning == "yes" else "no_reasoning"
    reuse_path = args.reuse_results or ROOT / "results/benchmark_250" / mode / "reuse" / f"qwen3-{args.model}" / "samples.jsonl"
    output_dir = args.output_dir or ROOT / "analysis/outputs" / f"deal_reuse_attention_comparison_{mode}_qwen3-{args.model}"
    if not args.benchmark_input.is_file() or not reuse_path.is_file():
        raise FileNotFoundError(f"Need benchmark input and reuse results: {args.benchmark_input}, {reuse_path}")
    directions = select_directions(load_jsonl(args.benchmark_input), load_jsonl(reuse_path))
    groups = {"correct": [row for row in directions if row["reuse_correct"]], "wrong": [row for row in directions if not row["reuse_correct"]]}
    if not groups["correct"] or not groups["wrong"]:
        raise ValueError("Both correct and wrong reuse groups must be nonempty")
    if args.max_per_group > 0:
        groups = {key: value[:args.max_per_group] for key, value in groups.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Selected Deal directions: correct={len(groups['correct'])}, wrong={len(groups['wrong'])}", flush=True)

    model_path = Path(snapshot_download(MODEL_IDS[args.model], local_files_only=not args.allow_download))
    validate_model_files(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="eager",
    ).eval()

    group_matrices: dict[str, dict[str, list[np.ndarray]]] = {
        group: {metric: [] for metric in METRICS} for group in groups
    }
    case_metrics: list[dict[str, Any]] = []
    total = sum(len(value) for value in groups.values())
    index = 0
    for group, cases in groups.items():
        for case in cases:
            index += 1
            parts = build_prompt_parts(
                tokenizer, case["record"], case["source_side"], enable_thinking=False,
                explicit_reasoning=args.reasoning == "yes", boxed_output=True,
            )
            ids = parts.full_ids.unsqueeze(0).to(model.device)
            length = ids.shape[1]
            positions = torch.arange(length, device=model.device, dtype=torch.long)
            with torch.inference_mode():
                output = model(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    position_ids=positions.unsqueeze(0), cache_position=positions,
                    output_attentions=True, use_cache=False, return_dict=True,
                )
            if output.attentions is None:
                raise RuntimeError("Model did not return attentions; eager attention is required")
            values = attention_masses(output.attentions, parts.block_token_start, parts.block_token_end)
            scalar_metrics = {}
            for metric in METRICS:
                normalized = resample_relative_positions(values[metric], args.bins)
                group_matrices[group][metric].append(normalized)
                scalar_metrics[f"mean_{metric}"] = float(np.nanmean(values[metric]))
            case_metrics.append({
                **{key: value for key, value in case.items() if key != "record"},
                "group": group,
                "source_prefix_tokens": int(parts.prefix_ids.numel()),
                "shared_block_tokens": int(parts.block_ids.numel()),
                **scalar_metrics,
            })
            print(f"[{index}/{total}] {group} {case['task_id']} {case['direction']} prefix-share={scalar_metrics['mean_prefix_share_of_history']:.3f}", flush=True)
            del output, ids, positions, values
            gc.collect(); torch.cuda.empty_cache()

    stacked = {group: {metric: np.stack(matrices) for metric, matrices in values.items()} for group, values in group_matrices.items()}
    np.savez_compressed(
        output_dir / "group_attention_matrices.npz",
        **{f"{group}_{metric}": matrix for group, metrics in stacked.items() for metric, matrix in metrics.items()},
    )
    plot_groups(output_dir / "group_attention_comparison.png", stacked)
    rng = np.random.default_rng(args.seed)
    summary: dict[str, Any] = {
        "model": args.model, "reasoning": args.reasoning, "attention_context": "source",
        "records": {group: len(cases) for group, cases in groups.items()}, "relative_position_bins": args.bins,
        "metrics": {},
    }
    for metric in METRICS:
        correct_values = np.array([row[f"mean_{metric}"] for row in case_metrics if row["group"] == "correct"])
        wrong_values = np.array([row[f"mean_{metric}"] for row in case_metrics if row["group"] == "wrong"])
        ci_low, ci_high = bootstrap_difference(correct_values, wrong_values, args.bootstrap_samples, rng)
        summary["metrics"][metric] = {
            "correct_mean": float(correct_values.mean()), "wrong_mean": float(wrong_values.mean()),
            "wrong_minus_correct": float(wrong_values.mean() - correct_values.mean()),
            "bootstrap_95_ci_wrong_minus_correct": [ci_low, ci_high],
        }
    (output_dir / "case_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in case_metrics), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
