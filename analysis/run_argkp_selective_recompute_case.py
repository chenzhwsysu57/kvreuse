#!/usr/bin/env python3
"""Compare full, raw KV reuse, and attention-selected partial recomputation.

This is a single-case diagnostic for ArgKP.  It reconstructs donor block KV
under ``--source-side``, relocates it into ``--target-side``, and then builds a
third cache token by token: selected block positions are forwarded under the
target cache, while every other position is appended from the relocated donor
KV.  It writes the exact generated text for all three conditions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_direct_reuse import (  # noqa: E402
    MODEL_IDS, build_prompt_parts, forward_ids, forward_suffix, greedy_continue,
    extract_cache_tensors, input_device, logits_comparison, make_dynamic_cache,
    relocate_block, result_from_generation, slice_cache,
    splice_prefix_block, validate_model_files,
)


DEFAULT_TASK = "argkp-dev-kp_15_3-kp_15_0-3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="1.7b", choices=sorted(MODEL_IDS))
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/argkp.jsonl")
    parser.add_argument("--task-id", default=DEFAULT_TASK)
    parser.add_argument("--source-side", choices=("a", "b"), default="a")
    parser.add_argument("--target-side", choices=("a", "b"), default="b")
    parser.add_argument(
        "--selection-file", type=Path,
        default=ROOT / "analysis/outputs/argkp_prefix_attention_qwen3-1.7b/selective_recompute_tokens.jsonl",
    )
    parser.add_argument("--max-spans", type=int, default=0,
                        help="use at most this many contiguous spans from the selection file; 0 uses every span")
    parser.add_argument("--recompute-prefix-tokens", type=int, default=0,
                        help="discard donor KV for the first N block tokens and recompute that contiguous prefix; overrides selection file")
    parser.add_argument("--drop-prefix-tokens", type=int, default=0,
                        help="do not put donor KV for the first N block tokens into the partial cache")
    parser.add_argument("--recompute-suffix-tokens", type=int, default=0,
                        help="recompute the final N block tokens under the target cache; may be used with or without --drop-prefix-tokens")
    parser.add_argument("--post-restatement", action="store_true",
                        help="insert a target-task restatement after the block and before the question")
    parser.add_argument("--reuse-only", action="store_true",
                        help="do not recompute or drop any block KV; useful for post-restatement-only control")
    parser.add_argument("--reasoning", choices=("no", "yes"), default="no",
                        help="use the benchmark's explicit visible-reasoning prompt")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="defaults to 128 without reasoning and 1024 with reasoning")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_record(path: Path, task_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("task_id") == task_id:
                    return record
    raise KeyError(f"{task_id} not found in {path}")


def load_selection(path: Path, task_id: str, side: str, max_spans: int) -> tuple[list[int], list[list[int]], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") == task_id and row.get("side") == side:
                spans = row.get("selected_spans")
                if not spans:
                    raise ValueError("selection file has no contiguous spans; rerun analyze_argkp_prefix_attention.py")
                spans = [[int(start), int(end)] for start, end in spans]
                if max_spans:
                    spans = spans[:max_spans]
                positions = [index for start, end in spans for index in range(start, end)]
                return positions, spans, row
    raise KeyError(f"no selection for {task_id} target side {side} in {path}")


def generate_result(
    model: Any, tokenizer: Any, logits: torch.Tensor, layers: list[tuple[torch.Tensor, torch.Tensor]],
    record: dict[str, Any], target: str, limit: int, elapsed: float, explicit_reasoning: bool,
    first_generated_position: int | None = None,
) -> dict[str, Any]:
    if first_generated_position is None:
        token_ids, hit_limit = greedy_continue(model, tokenizer, logits, layers, limit)
    else:
        token_ids, hit_limit = greedy_continue_at_position(
            model, tokenizer, logits, layers, first_generated_position, limit,
        )
    return result_from_generation(
        tokenizer, record["dataset"], record[f"gold_{target}"], token_ids, hit_limit,
        elapsed, limit, False, explicit_reasoning,
        "" if explicit_reasoning else "The answer is: \\boxed{",
    )


def target_record_with_post_restatement(record: dict[str, Any], target_side: str) -> dict[str, Any]:
    """Add a causal post-block bridge without changing donor block KV."""
    modified = dict(record)
    modified["question"] = (
        "Current task objective (takes priority): " + record[f"prefix_{target_side}"]
        + "\nImportant precaution: the preceding block and its cached states may contain signals "
        + "from other, unrelated task objectives. Ignore every such objective and use the preceding "
        + "candidate arguments only according to the current objective above.\n\n"
        + record["question"]
    )
    return modified


@torch.inference_mode()
def forward_suffix_at_positions(
    model: Any, past_layers: list[tuple[torch.Tensor, torch.Tensor]], suffix_ids: torch.Tensor,
    position_start: int,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Forward suffix while cache length and absolute RoPE positions differ.

    Dropping initial block KV makes the physical cache shorter, but later tokens
    must retain their original target-prompt RoPE positions.
    """
    device = input_device(model)
    past_length = past_layers[0][0].shape[-2]
    suffix = suffix_ids.unsqueeze(0).to(device)
    positions = torch.arange(position_start, position_start + suffix.shape[1], device=device, dtype=torch.long)
    cache = make_dynamic_cache(model, past_layers)
    output = model(
        input_ids=suffix,
        attention_mask=torch.ones((1, past_length + suffix.shape[1]), device=device, dtype=torch.long),
        position_ids=positions.unsqueeze(0), cache_position=positions,
        past_key_values=cache, use_cache=True, return_dict=True,
    )
    logits = output.logits[0, -1].detach().float()
    layers = extract_cache_tensors(output.past_key_values)
    return logits, layers


@torch.inference_mode()
def greedy_continue_at_position(
    model: Any, tokenizer: Any, initial_logits: torch.Tensor,
    prompt_layers: list[tuple[torch.Tensor, torch.Tensor]], first_position: int, max_new_tokens: int,
) -> tuple[list[int], bool]:
    device = input_device(model)
    cache = make_dynamic_cache(model, prompt_layers)
    physical_length = prompt_layers[0][0].shape[-2]
    current = initial_logits.argmax().view(1, 1).to(device)
    generated = [int(current.item())]
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    if generated[-1] in eos_ids:
        return generated, False
    for offset in range(1, max_new_tokens):
        position = torch.tensor([first_position + offset - 1], device=device, dtype=torch.long)
        output = model(
            input_ids=current,
            attention_mask=torch.ones((1, physical_length + 1), device=device, dtype=torch.long),
            position_ids=position.unsqueeze(0), cache_position=position,
            past_key_values=cache, use_cache=True, return_dict=True,
        )
        cache = output.past_key_values
        current = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(current.item()))
        physical_length += 1
    return generated, True


def main() -> int:
    args = parse_args()
    if args.source_side == args.target_side:
        raise ValueError("source and target sides must differ for a cross-prefix reuse diagnostic")
    limit = args.max_new_tokens if args.max_new_tokens is not None else (1024 if args.reasoning == "yes" else 128)
    if (args.max_spans < 0 or args.recompute_prefix_tokens < 0
            or args.drop_prefix_tokens < 0 or args.recompute_suffix_tokens < 0 or limit <= 0):
        raise ValueError("span counts must be non-negative and --max-new-tokens must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("selective recomputation requires CUDA")
    record = load_record(args.input, args.task_id)
    if args.recompute_prefix_tokens and (args.drop_prefix_tokens or args.recompute_suffix_tokens):
        raise ValueError("--recompute-prefix-tokens cannot be combined with drop/recompute-tail mode")
    if args.reuse_only and (args.recompute_prefix_tokens or args.drop_prefix_tokens or args.recompute_suffix_tokens):
        raise ValueError("--reuse-only cannot be combined with a KV modification")
    tail_requested = bool(args.drop_prefix_tokens or args.recompute_suffix_tokens)
    position_gap_mode = bool(args.drop_prefix_tokens)
    block_length = None  # checked once the prompt is rendered
    if args.reuse_only:
        selected, selected_spans, selection_row = [], [], None
    elif tail_requested:
        # The exact tail indices depend on the rendered block length below.
        selected, selected_spans, selection_row = [], [], None
    elif args.recompute_prefix_tokens:
        selected = list(range(args.recompute_prefix_tokens))
        selected_spans = [[0, args.recompute_prefix_tokens]]
        selection_row = None
    else:
        selected, selected_spans, selection_row = load_selection(
            args.selection_file, args.task_id, args.target_side, args.max_spans,
        )
    output_dir = args.output_dir or ROOT / "analysis/outputs" / f"selective_recompute_{args.task_id}_{args.source_side}_to_{args.target_side}_{args.reasoning}_reasoning"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(snapshot_download(MODEL_IDS[args.model], local_files_only=not args.allow_download))
    validate_model_files(model_path)
    print(f"Loading {MODEL_IDS[args.model]} from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa",
    ).eval()
    explicit_reasoning = args.reasoning == "yes"
    source = build_prompt_parts(tokenizer, record, args.source_side, enable_thinking=False, explicit_reasoning=explicit_reasoning, boxed_output=True)
    target_record = target_record_with_post_restatement(record, args.target_side) if args.post_restatement else record
    target = build_prompt_parts(tokenizer, target_record, args.target_side, enable_thinking=False, explicit_reasoning=explicit_reasoning, boxed_output=True)
    if not torch.equal(source.block_ids, target.block_ids):
        raise ValueError("source and target shared-block token IDs differ")
    block_length = int(target.block_ids.numel())
    if tail_requested:
        if args.drop_prefix_tokens >= block_length or args.recompute_suffix_tokens >= block_length:
            raise ValueError("drop/recompute lengths must be shorter than the block")
        if args.drop_prefix_tokens + args.recompute_suffix_tokens > block_length:
            raise ValueError("dropped prefix and recomputed suffix overlap")
        selected = list(range(block_length - args.recompute_suffix_tokens, block_length)) if args.recompute_suffix_tokens else []
        selected_spans = [[block_length - args.recompute_suffix_tokens, block_length]] if args.recompute_suffix_tokens else []
        selection_row = None
    if any(index < 0 or index >= target.block_ids.numel() for index in selected):
        raise ValueError("a selected token index lies outside the target block")

    # Dense target reference.
    started = time.perf_counter()
    full_logits, full_layers = forward_ids(model, target.full_ids)
    full = generate_result(model, tokenizer, full_logits, full_layers, record, args.target_side, limit, time.perf_counter() - started, explicit_reasoning)

    # Donor block KV, relocated to the target's absolute block positions.
    _, donor_layers = forward_ids(model, source.full_ids)
    donor_block = slice_cache(donor_layers, source.block_token_start, source.block_token_end)
    relocated = relocate_block(model, donor_block, source.block_token_start, target.block_token_start)

    # Raw direct reuse.
    _, target_prefix = forward_ids(model, target.prefix_ids)
    started = time.perf_counter()
    raw_layers = splice_prefix_block(target_prefix, relocated)
    raw_logits, raw_prompt_layers = forward_suffix(model, raw_layers, target.suffix_ids)
    raw = generate_result(model, tokenizer, raw_logits, raw_prompt_layers, record, args.target_side, limit, time.perf_counter() - started, explicit_reasoning)
    raw.update(logits_comparison(full_logits, raw_logits))

    # Selected block tokens are recomputed under the mixed target cache. In
    # tail mode, the first N donor tokens are absent altogether; this creates a
    # shorter physical cache, so all later forwards keep their original RoPE
    # positions explicitly.
    _, partial_layers = forward_ids(model, target.prefix_ids)
    selected_set = set(selected)
    dropped_set = set(range(args.drop_prefix_tokens)) if position_gap_mode else set()
    started = time.perf_counter()
    for index in range(target.block_ids.numel()):
        absolute_position = target.block_token_start + index
        if index in dropped_set:
            continue
        if index in selected_set:
            if position_gap_mode:
                _, partial_layers = forward_suffix_at_positions(
                    model, partial_layers, target.block_ids[index:index + 1], absolute_position,
                )
            else:
                _, partial_layers = forward_suffix(model, partial_layers, target.block_ids[index:index + 1])
        else:
            donor_token = slice_cache(relocated, index, index + 1)
            partial_layers = splice_prefix_block(partial_layers, donor_token)
    if position_gap_mode:
        partial_logits, partial_prompt_layers = forward_suffix_at_positions(
            model, partial_layers, target.suffix_ids, target.block_token_end,
        )
        partial = generate_result(
            model, tokenizer, partial_logits, partial_prompt_layers, record, args.target_side, limit,
            time.perf_counter() - started, explicit_reasoning, first_generated_position=int(target.full_ids.numel()),
        )
    else:
        partial_logits, partial_prompt_layers = forward_suffix(model, partial_layers, target.suffix_ids)
        partial = generate_result(model, tokenizer, partial_logits, partial_prompt_layers, record, args.target_side, limit, time.perf_counter() - started, explicit_reasoning)
    partial.update(logits_comparison(full_logits, partial_logits))

    selection_scores = {}
    if selection_row is not None:
        selection_scores = dict(zip(selection_row["selected_relative_block_token_indices"], selection_row["scores"]))
    selected_details = [
        {
            "relative_block_token_index": index,
            "token": tokenizer.decode([int(target.block_ids[index])]),
            "semantic_prefix_dependency": selection_scores.get(index),
        }
        for index in selected
    ]
    result = {
        "task_id": args.task_id, "source_side": args.source_side, "target_side": args.target_side,
        "reasoning": args.reasoning,
        "post_restatement": args.post_restatement,
        "dropped_prefix_block_tokens": args.drop_prefix_tokens,
        "target_gold": record[f"gold_{args.target_side}"],
        "block_tokens": int(target.block_ids.numel()),
        "selected_recomputed_spans": selected_spans,
        "selected_recomputed_tokens": selected_details,
        "full": full, "raw_reuse": raw, "selective_recompute": partial,
    }
    (output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = [
        f"# {args.task_id}: {args.source_side.upper()} → {args.target_side.upper()}",
        f"\nGold: `{result['target_gold']}`. Recomputed {len(selected)} of {result['block_tokens']} block BPE tokens in contiguous spans `{selected_spans}`.",
        "\n## Selected tokens\n",
        "| index | token | semantic prefix dependency |\n|---:|---|---:|",
        *[
            f"| {item['relative_block_token_index']} | `{item['token']}` | "
            f"{item['semantic_prefix_dependency']:.4f} |"
            if item["semantic_prefix_dependency"] is not None
            else f"| {item['relative_block_token_index']} | `{item['token']}` | fixed front-window |"
            for item in selected_details
        ],
    ]
    for name, value in (("Full recompute", full), ("Raw KV reuse", raw), ("Selective recompute", partial)):
        markdown.extend((f"\n## {name}\n", f"prediction: `{value['prediction']}`; correct: `{value['correct']}`\n", "```text", value["output_text"], "```"))
    (output_dir / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    for name, value in (("full", full), ("raw reuse", raw), ("selective", partial)):
        print(f"{name}: prediction={value['prediction']} correct={value['correct']} output={value['output_text']!r}", flush=True)
    print(f"Wrote {output_dir / 'comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
