#!/usr/bin/env python3
"""Measure how ArgKP block tokens attend to their prefix during dense prefill.

For each selected ArgKP record and prefix side, this script performs exactly one
eager-attention dense prefill.  For each layer ``l`` and shared-block token
``t`` it writes two complementary attention partitions:

``prefix_dependency[l, t]``
    Head-mean probability mass assigned to every key before the block.
``block_self_dependency[l, t]``
    Head-mean probability mass assigned to block keys through ``t`` itself.

They sum to one (up to floating-point rounding), because causal attention gives
the block query no other visible keys.  The query/key rotations are Qwen's
normal post-RoPE attention computation: the script asks the eager model for its
actual attention probabilities rather than reimplementing RoPE.

It also records a sparse/high-response diagnostic:

``peak_prefix_response[l, t] = max(head, prefix_key) attention(head, t, prefix_key)``.

Thus the first metric asks whether a query broadly depends on the prefix; the
second finds whether any individual prefix token receives exceptionally high
attention from any head.

For selective recomputation, the script also emits the same two measurements
restricted to the *semantic* role prefix: the changing proposition, PRO/CON
role, and key point.  Fixed system/chat-template tokens are excluded from that
restricted diagnostic, preventing attention sinks such as ``<|im_start|>``
from being mistaken for role dependence.
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
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_direct_reuse import MODEL_IDS, build_prompt_parts, load_jsonl, validate_model_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/argkp.jsonl")
    parser.add_argument("--sides", nargs="+", choices=("a", "b"), default=("a", "b"))
    parser.add_argument("--max-samples", type=int, default=0, help="0 analyses every ArgKP record")
    parser.add_argument("--selection-top-k", type=int, default=8,
                        help="top positions retained separately for semantic mass and semantic peak response")
    parser.add_argument("--recompute-span-length", type=int, default=8,
                        help="fixed window length used to cover each top-position set")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def attention_scores(
    attentions: tuple[torch.Tensor, ...], block_start: int, block_end: int, semantic_prefix_start: int,
) -> dict[str, np.ndarray]:
    """Return [layer, block-token] scores from eager causal attentions.

    ``attentions[layer]`` is [batch, heads, query, key].  For a block query,
    the visible keys partition into ``[:block_start]`` and
    ``[block_start:query_index + 1]``.  The two returned dependency scores are
    therefore a probability partition by construction.
    """
    if not (0 <= semantic_prefix_start < block_start < block_end):
        raise ValueError("the reusable block must have a nonempty preceding prefix")
    prefix_rows: list[np.ndarray] = []
    self_rows: list[np.ndarray] = []
    peak_rows: list[np.ndarray] = []
    peak_key_rows: list[np.ndarray] = []
    semantic_rows: list[np.ndarray] = []
    semantic_peak_rows: list[np.ndarray] = []
    semantic_peak_key_rows: list[np.ndarray] = []
    for layer_attention in attentions:
        weights = layer_attention[0].float()  # [heads, query, key]
        block_queries = weights[:, block_start:block_end, :]
        prefix_weights = block_queries[..., :block_start]
        # Average over heads only after summing every prefix key: this is the
        # ordinary total prefix-dependence score.
        prefix = prefix_weights.sum(dim=-1).mean(dim=0)
        # Every block query can see its own and preceding block tokens only.
        block = torch.stack([
            block_queries[:, relative, block_start:block_start + relative + 1].sum(dim=-1).mean()
            for relative in range(block_end - block_start)
        ])
        # For the high-response diagnostic retain the strongest individual
        # (head, prefix-key) edge, rather than summing or averaging it away.
        flattened = prefix_weights.permute(1, 0, 2).reshape(block_end - block_start, -1)
        peak, flat_index = flattened.max(dim=-1)
        peak_key = torch.remainder(flat_index, block_start)
        semantic_weights = block_queries[..., semantic_prefix_start:block_start]
        semantic = semantic_weights.sum(dim=-1).mean(dim=0)
        semantic_flattened = semantic_weights.permute(1, 0, 2).reshape(block_end - block_start, -1)
        semantic_peak, semantic_flat_index = semantic_flattened.max(dim=-1)
        semantic_peak_key = semantic_prefix_start + torch.remainder(
            semantic_flat_index, block_start - semantic_prefix_start
        )
        prefix_rows.append(prefix.cpu().numpy())
        self_rows.append(block.cpu().numpy())
        peak_rows.append(peak.cpu().numpy())
        peak_key_rows.append(peak_key.cpu().numpy())
        semantic_rows.append(semantic.cpu().numpy())
        semantic_peak_rows.append(semantic_peak.cpu().numpy())
        semantic_peak_key_rows.append(semantic_peak_key.cpu().numpy())
    prefix_array = np.stack(prefix_rows)
    self_array = np.stack(self_rows)
    return {
        "prefix_dependency": prefix_array,
        "block_self_dependency": self_array,
        "dependency_partition_error": np.abs(prefix_array + self_array - 1.0),
        "peak_prefix_response": np.stack(peak_rows),
        "peak_prefix_key_index": np.stack(peak_key_rows),
        "semantic_prefix_dependency": np.stack(semantic_rows),
        "semantic_peak_prefix_response": np.stack(semantic_peak_rows),
        "semantic_peak_prefix_key_index": np.stack(semantic_peak_key_rows),
    }


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def semantic_prefix_token_start(tokenizer: Any, parts: Any, record: dict[str, Any], side: str) -> int:
    """Locate the first token of the changing role prefix in the rendered chat prompt."""
    prefix = record[f"prefix_{side}"]
    char_start = parts.rendered.find(prefix)
    if char_start < 0:
        raise ValueError("role prefix was not preserved by the chat template")
    encoded = tokenizer(parts.rendered, add_special_tokens=False, return_offsets_mapping=True)
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if index >= parts.block_token_start:
            break
        if end > char_start:
            return index
    raise ValueError("could not find a token-aligned semantic role-prefix start")


def top_positions_and_covering_window(scores: np.ndarray, top_k: int, span_length: int) -> tuple[list[int], list[int]]:
    """Choose one fixed-length window covering as many top-score positions as possible."""
    if top_k <= 0 or span_length <= 0:
        return [], []
    length = min(span_length, len(scores))
    top = np.argsort(scores)[::-1][:min(top_k, len(scores))].astype(int).tolist()
    # Primary objective: how many selected positions are covered.  Ties go to
    # the window with the greatest selected-score sum, then total mass.
    candidates = []
    for start in range(len(scores) - length + 1):
        covered = [index for index in top if start <= index < start + length]
        candidates.append((len(covered), float(scores[covered].sum()) if covered else 0.0,
                           float(scores[start:start + length].sum()), -start, start))
    _, _, _, _, start = max(candidates)
    return top, [start, start + length]


def merge_spans(spans: list[list[int]]) -> list[list[int]]:
    """Return the contiguous union of overlapping or adjacent windows."""
    spans = sorted((start, end) for start, end in spans if end > start)
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def plot_dependency(case_dir: Path, scores: dict[str, np.ndarray], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = np.arange(scores["prefix_dependency"].shape[1])
    figure, axis = plt.subplots(figsize=(14, 4), constrained_layout=True)
    axis.plot(tokens, scores["prefix_dependency"].mean(axis=0), label="prefix dependency", color="#482878")
    axis.plot(tokens, scores["block_self_dependency"].mean(axis=0), label="block/self dependency", color="#22a884")
    axis.set(xlabel="Relative block token index", ylabel="Layer-mean attention mass", ylim=(0.0, 1.0), title=title)
    axis.legend(loc="best")
    figure.savefig(case_dir / "prefix_dependency_curve.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    panels = (("prefix_dependency", "Prefix dependency"), ("block_self_dependency", "Block/self dependency"))
    image = None
    for axis, (key, label) in zip(axes, panels):
        image = axis.imshow(scores[key], origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
        axis.set(ylabel="Layer", title=label)
    axes[-1].set_xlabel("Relative block token index")
    figure.colorbar(image, ax=axes, label="Attention mass", shrink=0.85)
    figure.suptitle(title)
    figure.savefig(case_dir / "prefix_dependency_heatmap.png", dpi=180)
    plt.close(figure)


def plot_peak_response(case_dir: Path, scores: dict[str, np.ndarray], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = np.arange(scores["peak_prefix_response"].shape[1])
    figure, axis = plt.subplots(figsize=(14, 4), constrained_layout=True)
    axis.plot(tokens, scores["peak_prefix_response"].mean(axis=0), label="layer mean", color="#f8961e")
    axis.plot(tokens, scores["peak_prefix_response"].max(axis=0), label="largest layer response", color="#d00000", alpha=0.8)
    axis.set(xlabel="Relative block token index", ylabel="Max single prefix-edge attention", ylim=(0.0, 1.0), title=title)
    axis.legend(loc="best")
    figure.savefig(case_dir / "peak_prefix_response_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    image = axis.imshow(scores["peak_prefix_response"], origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="magma")
    axis.set(xlabel="Relative block token index", ylabel="Layer", title=title)
    figure.colorbar(image, ax=axis, label="Max over head and prefix key")
    figure.savefig(case_dir / "peak_prefix_response_heatmap.png", dpi=180)
    plt.close(figure)


def plot_semantic_response(case_dir: Path, scores: dict[str, np.ndarray], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens = np.arange(scores["semantic_prefix_dependency"].shape[1])
    figure, axis = plt.subplots(figsize=(14, 4), constrained_layout=True)
    axis.plot(tokens, scores["semantic_prefix_dependency"].mean(axis=0), label="semantic prefix mass", color="#2166ac")
    axis.plot(tokens, scores["semantic_peak_prefix_response"].mean(axis=0), label="semantic peak (layer mean)", color="#b2182b")
    axis.set(xlabel="Relative block token index", ylabel="Attention score", ylim=(0.0, 1.0), title=title)
    axis.legend(loc="best")
    figure.savefig(case_dir / "semantic_prefix_response_curve.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    panels = (
        ("semantic_prefix_dependency", "Semantic role-prefix dependency"),
        ("semantic_peak_prefix_response", "Peak semantic role-prefix response"),
    )
    image = None
    for axis, (key, label) in zip(axes, panels):
        image = axis.imshow(scores[key], origin="lower", aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="cividis")
        axis.set(ylabel="Layer", title=label)
    axes[-1].set_xlabel("Relative block token index")
    figure.colorbar(image, ax=axes, label="Attention score", shrink=0.85)
    figure.suptitle(title)
    figure.savefig(case_dir / "semantic_prefix_response_heatmap.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Attention extraction requires a CUDA GPU node")
    if args.max_samples < 0 or args.selection_top_k <= 0 or args.recompute_span_length <= 0:
        raise ValueError("--max-samples must be non-negative; selection top-k and span length must be positive")
    records = [record for record in load_jsonl(args.input) if record.get("dataset") == "argkp"]
    if args.max_samples:
        records = records[:args.max_samples]
    if not records:
        raise ValueError("no ArgKP records selected")
    output_dir = args.output_dir or ROOT / "analysis/outputs" / f"argkp_prefix_attention_qwen3-{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(snapshot_download(MODEL_IDS[args.model], local_files_only=not args.allow_download))
    validate_model_files(model_path)
    print(f"Loading {MODEL_IDS[args.model]} from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # Eager is analysis-only: SDPA does not materialise attention probabilities.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="eager",
    ).eval()

    summaries: list[dict[str, Any]] = []
    selected_tokens: list[dict[str, Any]] = []
    total = len(records) * len(args.sides)
    completed = 0
    for record in records:
        for side in args.sides:
            completed += 1
            parts = build_prompt_parts(
                tokenizer, record, side, enable_thinking=False,
                explicit_reasoning=False, boxed_output=True,
            )
            semantic_start = semantic_prefix_token_start(tokenizer, parts, record, side)
            ids = parts.full_ids.unsqueeze(0).to(model.device)
            positions = torch.arange(ids.shape[1], device=model.device, dtype=torch.long)
            with torch.inference_mode():
                output = model(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    position_ids=positions.unsqueeze(0), cache_position=positions,
                    output_attentions=True, use_cache=False, return_dict=True,
                )
            if output.attentions is None:
                raise RuntimeError("Model did not return attentions; eager attention is required")
            scores = attention_scores(
                output.attentions, parts.block_token_start, parts.block_token_end, semantic_start,
            )
            case_dir = output_dir / f"{safe_name(record['task_id'])}__{side}"
            case_dir.mkdir(parents=True, exist_ok=True)
            title = f"{record['task_id']} | prefix {side}"
            np.savez_compressed(case_dir / "attention_scores.npz", **scores)
            plot_dependency(case_dir, scores, title)
            plot_peak_response(case_dir, scores, title)
            plot_semantic_response(case_dir, scores, title)
            semantic_score = scores["semantic_prefix_dependency"].mean(axis=0)
            semantic_peak = scores["semantic_peak_prefix_response"].mean(axis=0)
            semantic_top, semantic_window = top_positions_and_covering_window(
                semantic_score, args.selection_top_k, args.recompute_span_length,
            )
            peak_top, peak_window = top_positions_and_covering_window(
                semantic_peak, args.selection_top_k, args.recompute_span_length,
            )
            selected_spans = merge_spans([semantic_window, peak_window])
            selected_indices = [index for start, end in selected_spans for index in range(start, end)]
            selected_tokens.append({
                "task_id": record["task_id"], "side": side,
                "selection_score": "top semantic mass and top semantic peak response; each covered by one fixed window",
                "semantic_top_positions": semantic_top,
                "semantic_covering_window": semantic_window,
                "peak_response_top_positions": peak_top,
                "peak_response_covering_window": peak_window,
                "selected_spans": selected_spans,
                "selected_relative_block_token_indices": [int(index) for index in selected_indices],
                "tokens": [tokenizer.decode([int(parts.block_ids[index])]) for index in selected_indices],
                "scores": [float(semantic_score[index]) for index in selected_indices],
                "layer_mean_semantic_peak_response": [float(semantic_peak[index]) for index in selected_indices],
            })
            summary = {
                "task_id": record["task_id"], "side": side, "dataset": "argkp",
                "prefix_tokens": int(parts.prefix_ids.numel()), "block_tokens": int(parts.block_ids.numel()),
                "semantic_prefix_token_span": [int(semantic_start), int(parts.block_token_start)],
                "layers": int(scores["prefix_dependency"].shape[0]),
                "mean_prefix_dependency": float(scores["prefix_dependency"].mean()),
                "mean_peak_prefix_response": float(scores["peak_prefix_response"].mean()),
                "max_peak_prefix_response": float(scores["peak_prefix_response"].max()),
                "mean_semantic_prefix_dependency": float(scores["semantic_prefix_dependency"].mean()),
                "mean_semantic_peak_prefix_response": float(scores["semantic_peak_prefix_response"].mean()),
                "max_semantic_peak_prefix_response": float(scores["semantic_peak_prefix_response"].max()),
                "max_dependency_partition_error": float(scores["dependency_partition_error"].max()),
                "artifact_dir": str(case_dir),
            }
            summaries.append(summary)
            print(
                f"[{completed}/{total}] {record['task_id']} {side} "
                f"prefix={summary['mean_prefix_dependency']:.3f} "
                f"semantic={summary['mean_semantic_prefix_dependency']:.3f} "
                f"peak={summary['max_semantic_peak_prefix_response']:.3f}", flush=True,
            )
            del output, ids, positions, scores
            gc.collect()
            torch.cuda.empty_cache()
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "selective_recompute_tokens.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_tokens), encoding="utf-8",
    )
    print(f"Wrote {len(summaries)} ArgKP prefix analyses to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
