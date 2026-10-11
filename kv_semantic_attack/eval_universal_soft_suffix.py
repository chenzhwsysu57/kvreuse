#!/usr/bin/env python3
"""Evaluate a trained universal soft suffix on arbitrary, optionally unseen pools.

Run with ``torchrun`` for one independent model/evaluation shard per GPU.  Each
rank writes its own JSONL; rank zero verifies and merges all rows before
reporting Full, Direct, soft suffix, and repeated-target-prefix accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_direct_reuse as base  # noqa: E402
from scripts import batch_eval as batch_ops  # noqa: E402
from kv_semantic_attack.train_universal_soft_suffix import (  # noqa: E402
    read_records,
)
from kv_semantic_attack.universal_soft_suffix import (  # noqa: E402
    append_soft_suffix_batch,
    freeze_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--include-dataset", action="append", default=[],
                        help="dataset name to retain from inputs; repeat to form a union")
    parser.add_argument("--exclude-dataset", action="append", default=[],
                        help="dataset name to omit from inputs; repeat to form a union")
    parser.add_argument("--task-id-file", type=Path,
                        help="JSON list or manifest containing task_ids to evaluate")
    parser.add_argument("--exclude-task-id-file", type=Path,
                        help="JSON list or manifest containing task_ids to exclude")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records-per-input", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=12,
                        help="paired records per batch; each batch has twice as many A/B directions")
    parser.add_argument("--methods", nargs="+", choices=("full", "direct", "soft"), default=("soft",),
                        help="methods to generate; default soft avoids recomputing separately-run baselines")
    parser.add_argument("--eval-mode", choices=("no-reasoning", "reasoning"), default="no-reasoning",
                        help="render the evaluation prompt with or without an explicit reasoning request")
    parser.add_argument("--target-sides", nargs="+", choices=("a", "b"), default=("a", "b"))
    parser.add_argument("--profile-stages", action="store_true",
                        help="print CUDA-synchronized per-batch stage timings")
    parser.add_argument("--progress-accuracy", action="store_true",
                        help="show cumulative accuracy and parse rate after every evaluation batch")
    args = parser.parse_args()
    if args.max_records_per_input < 0 or args.max_new_tokens < 1 or args.batch_size < 1:
        parser.error("record limit must be non-negative and max-new-tokens positive")
    return args


def task_ids_from_file(path: Path, manifest_field: str = "dev_task_ids") -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        ids = value
    elif isinstance(value, dict) and isinstance(value.get(manifest_field), list):
        ids = value[manifest_field]
    else:
        raise ValueError(f"task-id file must be a JSON list or a split manifest with {manifest_field}")
    if not ids or not all(isinstance(task_id, str) and task_id for task_id in ids):
        raise ValueError("task-id file requires nonempty string IDs")
    return set(ids)


def rows(
    records: list[dict[str, Any]], target_sides: tuple[str, ...] | list[str] = ("a", "b")
) -> list[tuple[dict[str, Any], str]]:
    return [(record, side) for record in records for side in target_sides]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[row["dataset"]].append(row)

    def one(values: list[dict[str, Any]]) -> dict[str, Any]:
        methods = {}
        names = tuple(values[0]["methods"]) if values else ()
        for method in names:
            outcomes = [row["methods"][method] for row in values]
            methods[method] = {
                "correct": sum(bool(value["correct"]) for value in outcomes),
                "accuracy": sum(bool(value["correct"]) for value in outcomes) / len(outcomes),
                "parse_rate": sum(value["prediction"] is not None for value in outcomes) / len(outcomes),
            }
        return {"directions": len(values), "methods": methods}

    return {"overall": one(results), "by_dataset": {name: one(values) for name, values in sorted(groups.items())}}


@torch.inference_mode()
def _reuse_parts_batch(
    model: Any, tokenizer: Any, pairs: list[tuple[dict[str, Any], str]],
    *, soft: torch.Tensor | None, explicit_reasoning: bool, max_new_tokens: int,
    profile_stages: bool = False,
) -> list[dict[str, Any]]:
    """Batched Direct/Soft generation using batch_eval cache primitives."""
    timings: dict[str, float] = {}

    def measure(name: str, operation: Any) -> Any:
        if profile_stages:
            torch.cuda.synchronize(base.input_device(model))
        started = time.perf_counter()
        result = operation()
        if profile_stages:
            torch.cuda.synchronize(base.input_device(model))
            timings[name] = (time.perf_counter() - started) * 1000
        return result

    targets, sources = [], []
    def build_prompts() -> None:
        for record, target in pairs:
            source = "b" if target == "a" else "a"
            targets.append(base.build_prompt_parts(
                tokenizer, record, target, enable_thinking=False,
                explicit_reasoning=explicit_reasoning, boxed_output=True,
            ))
            sources.append(base.build_prompt_parts(
                tokenizer, record, source, enable_thinking=False,
                explicit_reasoning=explicit_reasoning, boxed_output=True,
            ))

    measure("prompt", build_prompts)
    _, source_layers = measure(
        "source_prefill", lambda: batch_ops.prefill_ids_batch(model, [part.full_ids for part in sources])
    )
    _, prefix_layers = measure(
        "target_prefix_prefill", lambda: batch_ops.prefill_ids_batch(model, [part.prefix_ids for part in targets])
    )

    def relocate_and_splice() -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        mixed = []
        for target, source, source_cache, prefix_cache in zip(targets, sources, source_layers, prefix_layers):
            source_block = base.slice_cache(source_cache, source.block_token_start, source.block_token_end)
            relocated = base.relocate_block(
                model, source_block, source.block_token_start, target.block_token_start
            )
            mixed.append(base.splice_prefix_block(prefix_cache, relocated))
        return mixed

    mixed = measure("rope_relocate_and_splice", relocate_and_splice)
    if soft is None:
        logits, caches = measure(
            "suffix_forward", lambda: batch_ops.forward_suffix_batch(
                model, mixed, [part.suffix_ids for part in targets]
            )
        )
        lengths = [int(cache[0][0].shape[-2]) for cache in caches]
    else:
        logits, caches, lengths = measure(
            "soft_suffix_forward", lambda: append_soft_suffix_batch(
                model, mixed, [part.suffix_ids for part in targets], soft
            )
        )
    padded = measure("decode_cache_pad", lambda: batch_ops.pad_cache(list(zip(caches, lengths)), max(lengths)))
    generated = measure(
        "decode", lambda: batch_ops.decode_batch(model, tokenizer, logits, padded, lengths, max_new_tokens)
    )
    output = []
    for (record, target), tokens in zip(pairs, generated):
        output.append(base.result_from_generation(
            tokenizer, record["dataset"], record[f"gold_{target}"], tokens,
            len(tokens) >= max_new_tokens, 0.0, max_new_tokens, False, explicit_reasoning,
            "" if explicit_reasoning else "The answer is: \\boxed{",
        ))
    if profile_stages:
        method = "soft" if soft is not None else "direct"
        timing_text = " ".join(f"{name}={value:.0f}ms" for name, value in timings.items())
        print(f"[profile {method}] directions={len(pairs)} {timing_text}", flush=True)
    return output


@torch.inference_mode()
def evaluate_generation_batch(
    model: Any, tokenizer: Any, soft: torch.Tensor, pair_rows: list[dict[str, Any]],
    target_sides: list[str], max_new_tokens: int, batch_size: int, explicit_reasoning: bool,
    methods: tuple[str, ...] | list[str], profile_stages: bool = False,
    progress_accuracy: bool = False,
) -> list[dict[str, Any]]:
    """Batch final selected-method evaluation; output schema matches prior evaluator."""
    all_rows: list[dict[str, Any]] = []
    correct = parsed = 0
    progress = tqdm(range(0, len(pair_rows), batch_size), desc="test", unit="batch", dynamic_ncols=True)
    for start in progress:
        records = pair_rows[start:start + batch_size]
        pairs = [(record, side) for record in records for side in target_sides]
        values: dict[str, list[dict[str, Any]]] = {}
        if "full" in methods:
            values["full"], _ = batch_ops.full_rows(
                model, tokenizer, records, "full", max_new_tokens, target_sides, explicit_reasoning
            )
        if "direct" in methods:
            values["direct"] = _reuse_parts_batch(
                model, tokenizer, pairs, soft=None, explicit_reasoning=explicit_reasoning,
                max_new_tokens=max_new_tokens, profile_stages=profile_stages,
            )
        if "soft" in methods:
            values["soft"] = _reuse_parts_batch(
                model, tokenizer, pairs, soft=soft, explicit_reasoning=explicit_reasoning,
                max_new_tokens=max_new_tokens, profile_stages=profile_stages,
            )
        for index, pair in enumerate(pairs):
            record, target = pair
            outcomes = {method: rows[index] for method, rows in values.items()}
            if len(outcomes) == 1:
                outcome = next(iter(outcomes.values()))
                correct += int(bool(outcome["correct"]))
                parsed += int(outcome["prediction"] is not None)
            all_rows.append({
                "task_id": record["task_id"], "dataset": record["dataset"],
                "target": target, "source": "b" if target == "a" else "a",
                "methods": outcomes,
            })
        if progress_accuracy and len(values) == 1:
            progress.set_postfix(
                directions=len(all_rows), accuracy=f"{correct / len(all_rows):.4f}",
                parse=f"{parsed / len(all_rows):.4f}",
            )
        else:
            progress.set_postfix(directions=len(all_rows))
    return all_rows


def main() -> int:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError(f"output directory is not empty: {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    soft = checkpoint.get("soft_prompt")
    if not isinstance(soft, torch.Tensor) or soft.ndim != 2:
        raise ValueError("checkpoint has no [tokens, hidden_size] soft_prompt")
    records, inputs, seen_ids = [], [], set()
    if args.task_id_file and args.exclude_task_id_file:
        raise ValueError("--task-id-file and --exclude-task-id-file cannot be combined")
    allowed_ids = task_ids_from_file(args.task_id_file) if args.task_id_file else None
    excluded_ids = (
        task_ids_from_file(args.exclude_task_id_file, "train_task_ids")
        if args.exclude_task_id_file else None
    )
    for path in args.input:
        selected = read_records(path)
        if args.include_dataset:
            selected = [record for record in selected if record["dataset"] in set(args.include_dataset)]
        if args.exclude_dataset:
            selected = [record for record in selected if record["dataset"] not in set(args.exclude_dataset)]
        if allowed_ids is not None:
            selected = [record for record in selected if record["task_id"] in allowed_ids]
        if excluded_ids is not None:
            selected = [record for record in selected if record["task_id"] not in excluded_ids]
        if args.max_records_per_input:
            selected = selected[:args.max_records_per_input]
        for record in selected:
            if record["task_id"] in seen_ids:
                raise ValueError(f"duplicate task_id across evaluation inputs: {record['task_id']}")
            seen_ids.add(record["task_id"])
        records.extend(selected)
        inputs.append({"path": str(path.resolve()), "records": len(selected),
                       "sha256": base.file_sha256(path)})
    all_rows = rows(records, args.target_sides)
    shard = all_rows[rank::world_size]
    if not shard:
        raise ValueError("evaluation has fewer directions than ranks")

    model_path = args.model_path.resolve()
    base.validate_model_files(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{local_rank}"}, attn_implementation="sdpa",
    )
    freeze_model(model)
    base.assert_default_rope(model)
    if soft.shape[1] != model.config.hidden_size:
        raise ValueError("checkpoint soft-prompt hidden size differs from model")
    soft = soft.to(device=base.input_device(model), dtype=torch.float32)
    shard_records = records[rank::world_size]
    report_rows = evaluate_generation_batch(
        model, tokenizer, soft, shard_records, list(args.target_sides),
        args.max_new_tokens, args.batch_size,
        explicit_reasoning=args.eval_mode == "reasoning", methods=tuple(args.methods),
        profile_stages=args.profile_stages, progress_accuracy=args.progress_accuracy,
    )
    part_path = args.output / f"rank_{rank:02d}.jsonl"
    base.write_jsonl_atomic(part_path, report_rows)
    if distributed:
        dist.barrier()
    if rank == 0:
        merged = []
        for other_rank in range(world_size):
            merged.extend(base.load_jsonl(args.output / f"rank_{other_rank:02d}.jsonl"))
        expected = {(record["task_id"], side) for record, side in all_rows}
        actual = {(row["task_id"], row["target"]) for row in merged}
        if actual != expected or len(merged) != len(actual):
            raise RuntimeError("merged evaluation rows do not exactly cover requested directions")
        merged.sort(key=lambda row: (row["dataset"], row["task_id"], row["target"]))
        base.write_jsonl_atomic(args.output / "samples.jsonl", merged)
        summary = {
            "complete": True,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_step": checkpoint.get("step"),
            "checkpoint_training_inputs": config.get("inputs"),
                "task_id_file": str(args.task_id_file.resolve()) if args.task_id_file else None,
                "exclude_task_id_file": str(args.exclude_task_id_file.resolve()) if args.exclude_task_id_file else None,
            "evaluation_inputs": inputs,
            "included_datasets": sorted(set(args.include_dataset)) or "all",
            "excluded_datasets": sorted(set(args.exclude_dataset)),
            "eval_mode": args.eval_mode,
            "world_size": world_size,
            "batch_size_pairs": args.batch_size,
            "methods": list(args.methods),
            "max_new_tokens": args.max_new_tokens,
            "target_sides": list(args.target_sides),
            **summarize(merged),
        }
        base.write_json_atomic(args.output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())