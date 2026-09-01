#!/usr/bin/env python3
"""Analyse prefix-vs-block attention for Deal RoPE-reuse hallucinations.

By default, a *hallucination* is a cross-reuse direction where the dense full
target answer is correct but RoPE-corrected direct reuse is wrong.  For every
such direction, this script runs one dense **source-prompt** prefill with eager
attention and measures, at every shared-block query token and every layer:

* attention mass on the source prefix (all keys before the shared block);
* attention mass on preceding shared-block tokens;
* prefix share among historical context, ``prefix / (prefix + prior_block)``.

The last quantity excludes self-attention, so it directly answers whether a
reused block token primarily reads its source prefix or its earlier block
context.  ``--attention-context target`` additionally supports the dense target
counterpart as a diagnostic control.
No generation is performed and this script never modifies experiment outputs.

Example (run in the kvreuse environment on a GPU node):

  /home/czw/miniconda3/envs/kvreuse/bin/python \
    analysis/plot_deal_prefix_attention.py --model 1.7b
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
    parser.add_argument("--reasoning", choices=("no", "yes"), default="no")
    parser.add_argument(
        "--attention-context", choices=("source", "target"), default="source",
        help="Analyse the source context that produced reused KV (default), or the dense target counterpart.",
    )
    parser.add_argument("--benchmark-input", type=Path, default=ROOT / "data/benchmark/benchmark_250.jsonl")
    parser.add_argument("--full-results", type=Path, default=None)
    parser.add_argument("--reuse-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=0, help="0 analyses every selected case.")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def defaults(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    mode = "reasoning" if args.reasoning == "yes" else "no_reasoning"
    base = ROOT / "results/benchmark_250" / mode
    full = args.full_results or base / "full" / f"qwen3-{args.model}" / "samples.jsonl"
    reuse = args.reuse_results or base / "reuse" / f"qwen3-{args.model}" / "samples.jsonl"
    output = args.output_dir or ROOT / "analysis/outputs" / f"deal_prefix_attention_{mode}_qwen3-{args.model}"
    return full, reuse, output


def select_cases(
    records: list[dict[str, Any]], full_rows: list[dict[str, Any]], reuse_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records_by_id = {row["task_id"]: row for row in records}
    full_by_id = {row["task_id"]: row for row in full_rows}
    selected: list[dict[str, Any]] = []
    directions = (("a_to_b", "b"), ("b_to_a", "a"))
    for reuse_row in reuse_rows:
        if reuse_row.get("dataset") != "deal_or_no_deal":
            continue
        task_id = reuse_row["task_id"]
        if task_id not in records_by_id or task_id not in full_by_id:
            raise KeyError(f"{task_id} missing from benchmark input or full results")
        for direction, target in directions:
            reuse = reuse_row.get("reuse", {}).get(direction)
            full = full_by_id[task_id].get("full", {}).get(target)
            if reuse is None or full is None:
                raise ValueError(
                    "Expected separate full-only and reuse-only runner outputs; "
                    f"{task_id} {direction} is incomplete."
                )
            if full["correct"] and not reuse["correct"]:
                selected.append({
                    "task_id": task_id,
                    "dataset": "deal_or_no_deal",
                    "direction": direction,
                    "source_side": "a" if direction == "a_to_b" else "b",
                    "target_side": target,
                    "gold": reuse["gold"],
                    "full_prediction": full["prediction"],
                    "full_output": full["output_text"],
                    "reuse_prediction": reuse["prediction"],
                    "reuse_output": reuse["output_text"],
                    "reuse_mean_key_cosine": reuse.get("mean_key_cosine"),
                    "reuse_mean_value_cosine": reuse.get("mean_value_cosine"),
                    "record": records_by_id[task_id],
                })
    return selected


def attention_masses(
    attentions: tuple[torch.Tensor, ...], block_start: int, block_end: int
) -> dict[str, np.ndarray]:
    """Return [layer, relative-block-token] attention partitions, head-averaged."""
    prefix_rows, prior_block_rows, block_with_self_rows = [], [], []
    for layer_attention in attentions:
        # [heads, query, key]; causal masking has already been applied by eager attention.
        weights = layer_attention[0].float()
        queries = weights[:, block_start:block_end, :]
        prefix_rows.append(queries[..., :block_start].sum(dim=-1).mean(dim=0).cpu().numpy())
        prior_values, inclusive_values = [], []
        for relative_index in range(block_end - block_start):
            query = queries[:, relative_index, :]
            prior_values.append(query[:, block_start:block_start + relative_index].sum(dim=-1).mean())
            inclusive_values.append(query[:, block_start:block_start + relative_index + 1].sum(dim=-1).mean())
        prior_block_rows.append(torch.stack(prior_values).cpu().numpy())
        block_with_self_rows.append(torch.stack(inclusive_values).cpu().numpy())
    prefix = np.stack(prefix_rows)
    prior = np.stack(prior_block_rows)
    inclusive = np.stack(block_with_self_rows)
    history_total = prefix + prior
    prefix_share = np.divide(prefix, history_total, out=np.full_like(prefix, np.nan), where=history_total > 0)
    return {
        "prefix_attention_mass": prefix,
        "prior_block_attention_mass": prior,
        "block_attention_mass_including_self": inclusive,
        "prefix_share_of_history": prefix_share,
    }


def plot_case(case_dir: Path, case: dict[str, Any], values: dict[str, np.ndarray], *, attention_context: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = (
        ("prefix_attention_mass", "Attention mass on pre-block prefix"),
        ("prior_block_attention_mass", "Attention mass on preceding block tokens"),
        ("prefix_share_of_history", "Prefix share: prefix / (prefix + preceding block)"),
    )
    figure, axes = plt.subplots(4, 1, figsize=(15, 14), constrained_layout=True)
    image = None
    for axis, (key, title) in zip(axes[:3], titles):
        matrix = values[key]
        image = axis.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_ylabel("Layer index")
    figure.colorbar(image, ax=axes[:3], label="Mean attention mass", shrink=0.85)

    token_index = np.arange(values["prefix_attention_mass"].shape[1])
    axes[3].plot(token_index, np.nanmean(values["prefix_attention_mass"], axis=0), label="pre-block prefix", color="#482878")
    axes[3].plot(token_index, np.nanmean(values["prior_block_attention_mass"], axis=0), label="preceding block", color="#22a884")
    axes[3].plot(token_index, np.nanmean(values["prefix_share_of_history"], axis=0), label="prefix share of history", color="#fde725")
    axes[3].set_xlim(0, max(1, len(token_index) - 1))
    axes[3].set_ylim(0.0, 1.0)
    axes[3].set_xlabel("Relative shared-block token index")
    axes[3].set_ylabel("Layer mean")
    axes[3].legend(loc="upper right")
    figure.suptitle(
        f"{case['task_id']} {case['direction']} | {attention_context} context | full={case['full_prediction']} reuse={case['reuse_prediction']} gold={case['gold']}",
        fontsize=12,
    )
    figure.savefig(case_dir / "attention_prefix_vs_block.png", dpi=170)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Attention extraction requires a CUDA GPU node")
    full_path, reuse_path, output_dir = defaults(args)
    for path in (args.benchmark_input, full_path, reuse_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cases = select_cases(load_jsonl(args.benchmark_input), load_jsonl(full_path), load_jsonl(reuse_path))
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    if not cases:
        raise ValueError("No full-correct/reuse-wrong Deal cases found")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_hallucinations.jsonl").write_text(
        "".join(json.dumps({key: value for key, value in case.items() if key != "record"}, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )

    model_path = Path(snapshot_download(MODEL_IDS[args.model], local_files_only=not args.allow_download))
    validate_model_files(model_path)
    print(f"Loading {MODEL_IDS[args.model]} from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # SDPA/Flash attention do not return an attention matrix.  Eager attention
    # is intentionally used here only for analysis; evaluation remains SDPA.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="eager",
    ).eval()

    summaries = []
    for index, case in enumerate(cases, 1):
        context_side = case["source_side"] if args.attention_context == "source" else case["target_side"]
        parts = build_prompt_parts(
            tokenizer, case["record"], context_side,
            enable_thinking=False,
            explicit_reasoning=args.reasoning == "yes",
            boxed_output=True,
        )
        ids = parts.full_ids.unsqueeze(0).to(model.device)
        length = ids.shape[1]
        positions = torch.arange(length, device=model.device, dtype=torch.long)
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        if output.attentions is None:
            raise RuntimeError("Model did not return attentions; eager attention is required")
        values = attention_masses(output.attentions, parts.block_token_start, parts.block_token_end)
        safe_name = f"{index:03d}_{case['task_id'].replace('/', '_')}__{case['direction']}"
        case_dir = output_dir / safe_name
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(case_dir / "attention_masses.npz", **values)
        plot_case(case_dir, case, values, attention_context=args.attention_context)
        summary = {
            **{key: value for key, value in case.items() if key != "record"},
            "attention_context": args.attention_context,
            "attention_context_side": context_side,
            "target_prefix_tokens": int(parts.prefix_ids.numel()),
            "shared_block_tokens": int(parts.block_ids.numel()),
            "suffix_tokens": int(parts.suffix_ids.numel()),
            "mean_prefix_attention_mass": float(np.nanmean(values["prefix_attention_mass"])),
            "mean_prior_block_attention_mass": float(np.nanmean(values["prior_block_attention_mass"])),
            "mean_prefix_share_of_history": float(np.nanmean(values["prefix_share_of_history"])),
            "artifact_dir": str(case_dir),
        }
        summaries.append(summary)
        print(
            f"[{index}/{len(cases)}] {case['task_id']} {case['direction']} "
            f"prefix-share={summary['mean_prefix_share_of_history']:.3f}", flush=True,
        )
        del output, ids, positions, values
        gc.collect()
        torch.cuda.empty_cache()
    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summaries)} cases to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
