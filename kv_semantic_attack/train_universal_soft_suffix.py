#!/usr/bin/env python3
"""Train one universal soft suffix to imitate Full under cross-prefix KV reuse.

Both A->B and B->A directions are trained with teacher-forced sequence KL.
Under ``torchrun``, each process owns one GPU and gradients for the shared soft
prompt are averaged synchronously before every optimizer update.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
from kv_semantic_attack.universal_soft_suffix import (  # noqa: E402
    _prefill_no_grad,
    append_suffix,
    build_reused_past,
    continuation_logits,
    forward_kl,
    freeze_model,
    full_next_logits,
    initialize_soft_prompt,
    selective_residual_kl,
    continuation_logits_batch,
    append_soft_suffix_batch,
)
from scripts import batch_eval as batch_ops  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="JSONL or JSON-list input; repeat for multiple task pools")
    parser.add_argument("--include-dataset", action="append", default=[],
                        help="dataset name to retain from inputs; repeat to form a union")
    parser.add_argument("--model", choices=tuple(base.MODEL_IDS), default="4b")
    parser.add_argument("--model-path", type=Path,
                        help="explicit local checkpoint path; bypass ModelScope cache lookup")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--explicit-reasoning", action="store_true",
                        help="train with the explicit reasoning prompt rather than answer-only assistant prefill")
    parser.add_argument("--training-prompt-modes", nargs="+", choices=("no-reasoning", "reasoning"),
                        help="prompt formats distilled into one shared soft prompt; overrides --explicit-reasoning")
    parser.add_argument("--soft-tokens", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=8,
                        help="directions per optimizer update after static KV/teacher precomputation")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--offset-threshold", type=float, default=1.0,
                        help="per-answer-token Full/Direct KL where repair weight is 0.5")
    parser.add_argument("--offset-scale", type=float, default=0.5,
                        help="positive sigmoid scale for the detached repair weight")
    parser.add_argument("--anchor-weight", type=float, default=1e-4,
                        help="coefficient of mean squared deviation from zero initialization")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--train-records-per-dataset", type=int, default=0,
                        help="use exactly this many source groups per dataset for training and hold out all others")
    parser.add_argument("--max-records-per-input", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-directions", type=int, default=16,
                        help="number of held-out directions for KL evaluation; 0 disables")
    parser.add_argument("--generate-directions", type=int, default=0,
                        help="held-out directions for final Full/Direct/Soft/Repeat generation")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--unseen-eval-input", type=Path, action="append", default=[],
                        help="JSONL pool for periodic unseen-task generation evaluation; repeat for multiple pools")
    parser.add_argument("--eval-pairs-per-task", type=int, default=0,
                        help="use this many held-out source groups per training task for periodic evaluation; takes precedence over --unseen-eval-input")
    parser.add_argument("--periodic-eval-kind", choices=("kl", "generation"), default="kl",
                        help="periodic hold-out metric: fast teacher-forced KL or free-generation accuracy")
    parser.add_argument("--unseen-eval-exclude-dataset", action="append", default=[],
                        help="dataset name excluded from periodic unseen-task evaluation; repeat for every training task")
    parser.add_argument("--unseen-eval-fraction", type=float, default=0.5,
                        help="deterministic source-group fraction retained per unseen task family")
    parser.add_argument("--unseen-eval-every", type=int, default=100,
                        help="evaluate unseen task accuracy every N steps; 0 disables periodic unseen evaluation")
    parser.add_argument("--unseen-eval-start-step", type=int, default=1,
                        help="first optimization step eligible for periodic unseen evaluation")
    parser.add_argument("--unseen-eval-patience", type=int, default=500,
                        help="stop after this many steps without an unseen Soft-accuracy improvement; 0 disables")
    parser.add_argument("--unseen-eval-mode", choices=("no-reasoning", "reasoning"), default="no-reasoning")
    parser.add_argument("--unseen-eval-max-new-tokens", type=int,
                        help="generation budget for periodic unseen evaluation; defaults to 32/no-reasoning or 4096/reasoning")
    parser.add_argument("--unseen-eval-batch-size", type=int, default=24,
                        help="directions per batched Soft-only unseen evaluation forward")
    parser.add_argument("--profile-step-timing", action="store_true",
                        help="write CUDA-synchronized per-stage timings to step_timings.jsonl")
    parser.add_argument("--save-final", action="store_true",
                        help="save final_latent_tokens.pt after the last optimization step")
    parser.add_argument("--teacher-tokens", type=int, default=8,
                        help="maximum Full greedy tokens distilled per direction")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="resume checkpoint.pt in the existing output directory")
    args = parser.parse_args()
    if (args.soft_tokens < 1 or args.steps < 1 or args.eval_every < 1
            or args.teacher_tokens < 1 or args.train_batch_size < 1):
        parser.error("soft-tokens, steps, eval-every, and teacher-tokens must be positive")
    if (args.learning_rate <= 0 or args.temperature <= 0 or args.grad_clip <= 0
            or args.offset_scale <= 0 or args.anchor_weight < 0):
        parser.error("learning-rate, temperature, grad-clip, offset-scale must be positive; anchor-weight non-negative")
    if not 0 < args.train_fraction < 1:
        parser.error("train-fraction must be between zero and one")
    if (args.max_records_per_input < 0 or args.eval_directions < 0
            or args.generate_directions < 0 or args.train_records_per_dataset < 0
            or args.unseen_eval_every < 0 or args.unseen_eval_patience < 0):
        parser.error("record and evaluation limits must be non-negative")
    if args.eval_pairs_per_task < 0:
        parser.error("eval-pairs-per-task must be non-negative")
    if args.unseen_eval_start_step < 1:
        parser.error("unseen-eval-start-step must be positive")
    if not 0 < args.unseen_eval_fraction <= 1:
        parser.error("unseen-eval-fraction must be in (0, 1]")
    if args.unseen_eval_max_new_tokens is not None and args.unseen_eval_max_new_tokens < 1:
        parser.error("unseen-eval-max-new-tokens must be positive")
    if args.unseen_eval_batch_size < 1:
        parser.error("unseen-eval-batch-size must be positive")
    if args.training_prompt_modes and len(set(args.training_prompt_modes)) != len(args.training_prompt_modes):
        parser.error("training-prompt-modes must not repeat a mode")
    return args


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"JSON input must be a list: {path}")
        records = value
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    required = {"task_id", "dataset", "prefix_a", "prefix_b", "shared_block",
                "question", "gold_a", "gold_b"}
    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{path}:{index}: missing fields {sorted(missing)}")
        if record["gold_a"] == record["gold_b"]:
            raise ValueError(f"{path}:{index}: identical A/B gold")
    return records


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def group_key(record: dict[str, Any]) -> str:
    for key in ("shared_data_id", "source_conversation_id", "source_claim_id", "task_id"):
        value = record.get(key)
        if value is None and isinstance(record.get("metadata"), dict):
            value = record["metadata"].get(key)
        if value is not None:
            return f"{record['dataset']}:{value}"
    return f"{record['dataset']}:{record['task_id']}"


def split_records(
    records: list[dict[str, Any]], train_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split source groups within every dataset, preserving both directions."""
    datasets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        datasets[record["dataset"]][group_key(record)].append(record)
    train, dev = [], []
    for dataset_index, dataset in enumerate(sorted(datasets)):
        groups = datasets[dataset]
        keys = sorted(groups)
        random.Random(seed + dataset_index).shuffle(keys)
        if len(keys) < 2:
            raise ValueError(
                f"dataset {dataset!r} needs at least two source groups for train/dev"
            )
        train_count = min(len(keys) - 1, max(1, round(len(keys) * train_fraction)))
        train_keys = set(keys[:train_count])
        train.extend(
            record for key in keys if key in train_keys for record in groups[key]
        )
        dev.extend(
            record for key in keys if key not in train_keys for record in groups[key]
        )
    return train, dev


def fixed_train_records(
    records: list[dict[str, Any]], count: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use a small fixed number of source groups per dataset for training.

    This is intentionally unlike the ordinary 80/20 split: all remaining
    groups form a large untouched hold-out test pool, not an optimization dev
    set. The caller must keep it out of training and checkpoint selection.
    """
    if count < 1:
        raise ValueError("fixed training count must be positive")
    datasets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        datasets[record["dataset"]][group_key(record)].append(record)
    train, holdout = [], []
    for dataset_index, dataset in enumerate(sorted(datasets)):
        groups = datasets[dataset]
        keys = sorted(groups)
        if len(keys) <= count:
            raise ValueError(
                f"dataset {dataset!r} needs more than {count} source groups for hold-out"
            )
        random.Random(seed + dataset_index).shuffle(keys)
        train_keys = set(keys[:count])
        train.extend(record for key in keys if key in train_keys for record in groups[key])
        holdout.extend(record for key in keys if key not in train_keys for record in groups[key])
    return train, holdout


def deterministic_fraction_per_dataset(
    records: list[dict[str, Any]], fraction: float, seed: int
) -> list[dict[str, Any]]:
    """Retain a deterministic fraction of source groups separately per dataset."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["dataset"]][group_key(record)].append(record)
    selected: list[dict[str, Any]] = []
    for index, dataset in enumerate(sorted(grouped)):
        groups = grouped[dataset]
        keys = sorted(groups)
        random.Random(seed + index).shuffle(keys)
        take = max(1, round(len(keys) * fraction))
        selected.extend(record for key in keys[:take] for record in groups[key])
    return selected


def fixed_records_per_dataset(
    records: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Select exactly ``count`` held-out source groups per dataset deterministically."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["dataset"]][group_key(record)].append(record)
    selected: list[dict[str, Any]] = []
    for index, dataset in enumerate(sorted(grouped)):
        groups = grouped[dataset]
        keys = sorted(groups)
        if len(keys) < count:
            raise ValueError(
                f"held-out dataset {dataset!r} has {len(keys)} groups, fewer than eval-pairs-per-task={count}"
            )
        random.Random(seed + index).shuffle(keys)
        selected.extend(record for key in keys[:count] for record in groups[key])
    return selected


def directions(
    records: list[dict[str, Any]], explicit_reasoning: bool | None = None
) -> list[tuple[dict[str, Any], str] | tuple[dict[str, Any], str, bool]]:
    """Expand records into directions, optionally tagging their prompt format."""
    if explicit_reasoning is None:
        return [(record, side) for record in records for side in ("a", "b")]
    return [(record, side, explicit_reasoning) for record in records for side in ("a", "b")]


def task_family(record: dict[str, Any]) -> str:
    """Use synthetic rule types as groups; otherwise use the dataset name."""
    attack_type = record.get("attack_type")
    if not attack_type and isinstance(record.get("metadata"), dict):
        attack_type = record["metadata"].get("attack_type")
    if isinstance(attack_type, str) and attack_type:
        return f"synthetic:{attack_type}"
    return f"dataset:{record['dataset']}"


def balanced_row_pools(
    rows: list[tuple[dict[str, Any], str]], seed: int
) -> dict[str, list[tuple[dict[str, Any], str]]]:
    pools: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for row in rows:
        pools[task_family(row[0])].append(row)
    for index, family in enumerate(sorted(pools)):
        random.Random(seed + index).shuffle(pools[family])
    return dict(pools)


def balanced_row(
    pools: dict[str, list[tuple[dict[str, Any], str]]], step: int, rank: int, world_size: int
) -> tuple[dict[str, Any], str]:
    """Cycle task families uniformly, then cycle deterministic rows within a family."""
    families = sorted(pools)
    slot = (step - 1) * world_size + rank
    family = families[slot % len(families)]
    return pools[family][(slot // len(families)) % len(pools[family])]


def balanced_batch(
    pools: dict[str, list[tuple[dict[str, Any], str]]], step: int, batch_size: int
) -> list[tuple[dict[str, Any], str]]:
    return [balanced_row(pools, (step - 1) * batch_size + offset + 1, 0, 1)
            for offset in range(batch_size)]


def precompute_training_states(
    model: Any, tokenizer: Any, rows: list[tuple[dict[str, Any], str, bool]], teacher_tokens: int,
    batch_size: int, show_progress: bool = False, explicit_reasoning: bool = False,
) -> dict[tuple[str, str, bool], tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]]:
    """Batch precompute all frozen Full/Direct/KV states once before optimization."""
    states = {}
    starts = range(0, len(rows), batch_size)
    if show_progress:
        starts = tqdm(starts, total=(len(rows) + batch_size - 1) // batch_size,
                      desc="precompute", unit="batch", dynamic_ncols=True)
    for start in starts:
        chunk = rows[start:start + batch_size]
        targets = [prompt_parts(tokenizer, record, side, row_reasoning) for record, side, row_reasoning in chunk]
        sources = [prompt_parts(tokenizer, record, "b" if side == "a" else "a", row_reasoning)
               for record, side, row_reasoning in chunk]
        full_logits, full_layers = batch_ops.prefill_ids_batch(model, [part.full_ids for part in targets])
        full_lengths = [int(part.full_ids.numel()) for part in targets]
        padded_full_layers = batch_ops.pad_cache(
            list(zip(full_layers, full_lengths)), max(full_lengths)
        )
        generated = batch_ops.decode_batch(
            model, tokenizer, full_logits, padded_full_layers, full_lengths, teacher_tokens,
        )
        _, source_layers = batch_ops.prefill_ids_batch(model, [part.full_ids for part in sources])
        _, target_prefix_layers = batch_ops.prefill_ids_batch(model, [part.prefix_ids for part in targets])
        reused = []
        answers = []
        full_pasts = []
        for index, (target, source) in enumerate(zip(targets, sources)):
            answer = torch.tensor(generated[index], dtype=torch.long, device=base.input_device(model))
            answers.append(answer)
            full_pasts.append(base.slice_cache(full_layers[index], 0, target.block_token_end))
            source_block = base.slice_cache(
                source_layers[index], source.block_token_start, source.block_token_end
            )
            relocated = base.relocate_block(
                model, source_block, source.block_token_start, target.block_token_start
            )
            reused.append(base.splice_prefix_block(target_prefix_layers[index], relocated))
        teacher_rows = [(past, target.suffix_ids, answer)
                        for past, target, answer in zip(full_pasts, targets, answers)]
        direct_rows = [(past, target.suffix_ids, answer)
                       for past, target, answer in zip(reused, targets, answers)]
        teacher_logits = continuation_logits_batch(model, teacher_rows, None)
        direct_logits = continuation_logits_batch(model, direct_rows, None)
        for index, ((record, side, row_reasoning), target, answer, past, teacher, direct) in enumerate(
            zip(chunk, targets, answers, reused, teacher_logits, direct_logits)
        ):
            states[(record["task_id"], side, row_reasoning)] = (
                teacher.detach(), direct.detach(), past, target.suffix_ids.detach(), answer.detach()
            )
    return states


@torch.no_grad()
def evaluate_soft_generation_batch(
    model: Any, tokenizer: Any, soft_prompt: torch.Tensor,
    rows: list[tuple[dict[str, Any], str]], max_new_tokens: int, batch_size: int,
    explicit_reasoning: bool, profile_stages: bool = False,
) -> dict[str, float]:
    """Fast early-stop metric: Soft-only generation in cache-aware batches."""
    correct = completed = parsed = 0
    started = time.perf_counter()
    total_batches = (len(rows) + batch_size - 1) // batch_size
    def cuda_stage(operation: Any) -> tuple[Any, float]:
        if profile_stages:
            torch.cuda.synchronize(base.input_device(model))
        stage_started = time.perf_counter()
        value = operation()
        if profile_stages:
            torch.cuda.synchronize(base.input_device(model))
        return value, (time.perf_counter() - stage_started) * 1000

    for start in range(0, len(rows), batch_size):
        batch_started = time.perf_counter()
        chunk = rows[start:start + batch_size]
        prompt_started = time.perf_counter()
        targets = [prompt_parts(tokenizer, record, target, explicit_reasoning)
                   for record, target in chunk]
        sources = [prompt_parts(tokenizer, record, "b" if target == "a" else "a", explicit_reasoning)
                   for record, target in chunk]
        prompt_ms = (time.perf_counter() - prompt_started) * 1000
        (_, source_layers), source_prefill_ms = cuda_stage(
            lambda: batch_ops.prefill_ids_batch(model, [part.full_ids for part in sources])
        )
        (_, prefix_layers), prefix_prefill_ms = cuda_stage(
            lambda: batch_ops.prefill_ids_batch(model, [part.prefix_ids for part in targets])
        )
        def relocate() -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
            mixed = []
            for target, source, source_cache, prefix_cache in zip(targets, sources, source_layers, prefix_layers):
                source_block = base.slice_cache(
                    source_cache, source.block_token_start, source.block_token_end
                )
                relocated = base.relocate_block(
                    model, source_block, source.block_token_start, target.block_token_start
                )
                mixed.append(base.splice_prefix_block(prefix_cache, relocated))
            return mixed
        mixed, relocate_ms = cuda_stage(relocate)
        (soft_output, soft_prefill_ms) = cuda_stage(
            lambda: append_soft_suffix_batch(model, mixed, [part.suffix_ids for part in targets], soft_prompt)
        )
        logits, caches, lengths = soft_output
        def decode() -> list[list[int]]:
            padded_caches = batch_ops.pad_cache(list(zip(caches, lengths)), max(lengths))
            return batch_ops.decode_batch(model, tokenizer, logits, padded_caches, lengths, max_new_tokens)
        generated, decode_ms = cuda_stage(decode)
        mean_generated_tokens = sum(len(tokens) for tokens in generated) / len(generated)
        parse_started = time.perf_counter()
        for (record, target), tokens in zip(chunk, generated):
            result = base.result_from_generation(
                tokenizer, record["dataset"], record[f"gold_{target}"], tokens,
                False, 0.0, max_new_tokens, False, False, "The answer is: \\boxed{"
            )
            completed += 1
            parsed += int(result["prediction"] is not None)
            correct += int(bool(result["correct"]))
        parse_ms = (time.perf_counter() - parse_started) * 1000
        batch_index = start // batch_size + 1
        print(
            f"[unseen-eval {batch_index}/{total_batches}] directions={completed}/{len(rows)} "
            f"batch_elapsed={time.perf_counter() - batch_started:.3f}s "
            f"prompt={prompt_ms:.0f}ms source={source_prefill_ms:.0f}ms "
            f"prefix={prefix_prefill_ms:.0f}ms rope={relocate_ms:.0f}ms "
            f"soft={soft_prefill_ms:.0f}ms decode={decode_ms:.0f}ms parse={parse_ms:.0f}ms "
            f"mean_generated_tokens={mean_generated_tokens:.1f} "
            f"total_elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return {
        "soft_accuracy": correct / completed if completed else 0.0,
        "soft_parse_rate": parsed / completed if completed else 0.0,
        "directions": completed,
        "elapsed_seconds": time.perf_counter() - started,
    }


def prompt_parts(
    tokenizer: Any, record: dict[str, Any], side: str, explicit_reasoning: bool = False
) -> base.PromptParts:
    return base.build_prompt_parts(
        tokenizer, record, side, enable_thinking=False,
        explicit_reasoning=explicit_reasoning, boxed_output=True,
    )


def direction_tensors(
    model: Any, tokenizer: Any, record: dict[str, Any], target: str
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    source = "b" if target == "a" else "a"
    target_parts = prompt_parts(tokenizer, record, target)
    source_parts = prompt_parts(tokenizer, record, source)
    teacher = full_next_logits(model, target_parts.full_ids)
    reused = build_reused_past(model, target_parts, source_parts)
    return teacher, reused, target_parts.suffix_ids


def direction_sequence_tensors(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    target: str,
    teacher_tokens: int,
    timings_ms: dict[str, float] | None = None,
    explicit_reasoning: bool = False,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Build Full sequence targets and reused student state for one direction."""
    source = "b" if target == "a" else "a"
    target_parts = prompt_parts(tokenizer, record, target, explicit_reasoning)
    source_parts = prompt_parts(tokenizer, record, source, explicit_reasoning)
    def measure(name: str, operation: Any) -> Any:
        if timings_ms is None:
            return operation()
        torch.cuda.synchronize(base.input_device(model))
        started = time.perf_counter()
        result = operation()
        torch.cuda.synchronize(base.input_device(model))
        timings_ms[name] = (time.perf_counter() - started) * 1000
        return result

    full_logits, full_layers = measure(
        "full_prefill", lambda: _prefill_no_grad(model, target_parts.full_ids)
    )
    answer_tokens, _ = measure(
        "teacher_greedy_decode", lambda: base.greedy_continue(
            model, tokenizer, full_logits, full_layers, teacher_tokens
        )
    )
    answer = torch.tensor(answer_tokens, dtype=torch.long)
    full_block_past = base.slice_cache(full_layers, 0, target_parts.block_token_end)
    def teacher_continuation() -> torch.Tensor:
        with torch.no_grad():
            return continuation_logits(
                model, full_block_past, target_parts.suffix_ids, answer, None
            ).detach()

    teacher = measure("teacher_continuation", teacher_continuation)
    reused = build_reused_past(model, target_parts, source_parts, timings_ms)
    return teacher, reused, target_parts.suffix_ids, answer


def evaluate_kl(
    model: Any,
    tokenizer: Any,
    soft_prompt: torch.Tensor,
    rows: list[tuple[dict[str, Any], str] | tuple[dict[str, Any], str, bool]],
    temperature: float,
    teacher_tokens: int,
    offset_threshold: float,
    offset_scale: float,
    explicit_reasoning: bool = False,
) -> dict[str, float]:
    direct_losses, soft_losses, preservation_losses, repair_weights = [], [], [], []
    direct_agreements, soft_agreements = [], []
    for row in rows:
        record, target = row[:2]
        row_reasoning = row[2] if len(row) == 3 else explicit_reasoning
        teacher, reused, suffix_ids, answer = direction_sequence_tensors(
            model, tokenizer, record, target, teacher_tokens,
            explicit_reasoning=row_reasoning,
        )
        with torch.no_grad():
            direct = continuation_logits(model, reused, suffix_ids, answer, None)
            soft = continuation_logits(model, reused, suffix_ids, answer, soft_prompt)
            direct_losses.append(float(forward_kl(teacher, direct, temperature=temperature)
                                       / answer.numel()))
            soft_losses.append(float(forward_kl(teacher, soft, temperature=temperature)
                                     / answer.numel()))
            _, weight, _, direct_to_soft = selective_residual_kl(
                teacher, direct, soft, temperature=temperature,
                offset_threshold=offset_threshold * answer.numel(),
                offset_scale=offset_scale * answer.numel(),
            )
            repair_weights.append(float(weight))
            preservation_losses.append(float(direct_to_soft / answer.numel()))
            teacher_top = teacher.argmax(dim=-1)
            direct_agreements.extend((direct.argmax(dim=-1) == teacher_top).tolist())
            soft_agreements.extend((soft.argmax(dim=-1) == teacher_top).tolist())
    return {
        "directions": len(rows),
        "direct_kl": sum(direct_losses) / len(direct_losses),
        "soft_kl": sum(soft_losses) / len(soft_losses),
        "soft_vs_direct_kl": sum(preservation_losses) / len(preservation_losses),
        "mean_repair_weight": sum(repair_weights) / len(repair_weights),
        "answer_tokens": len(direct_agreements),
        "direct_teacher_token_agreement": sum(direct_agreements) / len(direct_agreements),
        "soft_teacher_token_agreement": sum(soft_agreements) / len(soft_agreements),
    }


def generation_result(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    target: str,
    logits: torch.Tensor,
    layers: list[tuple[torch.Tensor, torch.Tensor]],
    max_new_tokens: int,
) -> dict[str, Any]:
    tokens, hit_limit = base.greedy_continue(
        model, tokenizer, logits.detach(), layers, max_new_tokens
    )
    return base.result_from_generation(
        tokenizer, record["dataset"], record[f"gold_{target}"], tokens,
        hit_limit, 0.0, max_new_tokens, False, False, "The answer is: \\boxed{",
    )


def evaluate_generation(
    model: Any,
    tokenizer: Any,
    soft_prompt: torch.Tensor,
    rows: list[tuple[dict[str, Any], str]],
    max_new_tokens: int,
    explicit_reasoning: bool = False,
) -> dict[str, Any]:
    results = []
    for record, target in rows:
        source = "b" if target == "a" else "a"
        target_parts = prompt_parts(tokenizer, record, target, explicit_reasoning)
        source_parts = prompt_parts(tokenizer, record, source, explicit_reasoning)
        _, full_layers = _prefill_no_grad(model, target_parts.full_ids)
        full_logits = full_next_logits(model, target_parts.full_ids)
        reused = build_reused_past(model, target_parts, source_parts)
        with torch.no_grad():
            direct_logits, direct_layers = append_suffix(
                model, reused, target_parts.suffix_ids, None, return_cache=True
            )
            soft_logits, soft_layers = append_suffix(
                model, reused, target_parts.suffix_ids, soft_prompt, return_cache=True
            )
            repeated = base.with_repeated_target_prefix(record, target)
            repeat_parts = prompt_parts(tokenizer, repeated, target, explicit_reasoning)
            repeat_logits, repeat_layers = append_suffix(
                model, reused, repeat_parts.suffix_ids, None, return_cache=True
            )
        assert direct_layers is not None and soft_layers is not None and repeat_layers is not None
        methods = {
            "full": generation_result(model, tokenizer, record, target, full_logits,
                                      full_layers, max_new_tokens),
            "direct": generation_result(model, tokenizer, record, target, direct_logits,
                                        direct_layers, max_new_tokens),
            "soft": generation_result(model, tokenizer, record, target, soft_logits,
                                      soft_layers, max_new_tokens),
            "repeat_prefix": generation_result(model, tokenizer, record, target, repeat_logits,
                                               repeat_layers, max_new_tokens),
        }
        results.append({"task_id": record["task_id"], "dataset": record["dataset"],
                        "target": target, "source": source, "methods": methods})
    summary: dict[str, Any] = {"directions": len(results), "methods": {}}
    for method in ("full", "direct", "soft", "repeat_prefix"):
        values = [row["methods"][method] for row in results]
        summary["methods"][method] = {
            "correct": sum(bool(value["correct"]) for value in values),
            "accuracy": sum(bool(value["correct"]) for value in values) / len(values),
            "parse_rate": sum(value["prediction"] is not None for value in values) / len(values),
        }
    return {"summary": summary, "results": results}


def save_checkpoint(
    path: Path, soft_prompt: torch.Tensor, optimizer: torch.optim.Optimizer,
    step: int, config: dict[str, Any], history: list[dict[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"soft_prompt": soft_prompt.detach().cpu(), "optimizer": optimizer.state_dict(),
                "step": step, "config": config, "history": history}, temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for universal soft-suffix training")
    if rank == 0 and not args.resume and args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output}")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    records: list[dict[str, Any]] = []
    inputs = []
    seen_ids = set()
    for path in args.input:
        selected = read_records(path)
        if args.include_dataset:
            allowed = set(args.include_dataset)
            selected = [record for record in selected if record["dataset"] in allowed]
            if not selected:
                raise ValueError(f"{path}: no records match --include-dataset")
        if args.max_records_per_input:
            selected = selected[:args.max_records_per_input]
        for record in selected:
            if record["task_id"] in seen_ids:
                raise ValueError(f"duplicate task_id across inputs: {record['task_id']}")
            seen_ids.add(record["task_id"])
        records.extend(selected)
        inputs.append({"path": str(path.resolve()), "sha256": file_sha256(path),
                       "records": len(selected)})
    if args.train_records_per_dataset:
        train_records, dev_records = fixed_train_records(
            records, args.train_records_per_dataset, args.seed
        )
        split_description = (
            f"fixed {args.train_records_per_dataset} source groups per dataset for training; "
            "all remaining groups are untouched hold-out test data"
        )
    else:
        train_records, dev_records = split_records(records, args.train_fraction, args.seed)
        split_description = "within-task-family source-group split; not an unseen-task evaluation"
    training_prompt_modes = args.training_prompt_modes or (
        ["reasoning"] if args.explicit_reasoning else ["no-reasoning"]
    )
    train_rows = [
        row
        for mode in training_prompt_modes
        for row in directions(train_records, explicit_reasoning=mode == "reasoning")
    ]
    dev_rows = [
        row
        for mode in training_prompt_modes
        for row in directions(dev_records, explicit_reasoning=mode == "reasoning")
    ]
    random.Random(args.seed + 2).shuffle(dev_rows)
    train_pools = balanced_row_pools(train_rows, args.seed + 1)
    holdout_eval_records = (
        fixed_records_per_dataset(dev_records, args.eval_pairs_per_task, args.seed + 4)
        if args.eval_pairs_per_task else []
    )
    holdout_eval_rows = directions(holdout_eval_records)
    unseen_records: list[dict[str, Any]] = []
    unseen_seen_ids: set[str] = set()
    excluded_unseen = set(args.unseen_eval_exclude_dataset)
    for path in args.unseen_eval_input:
        for record in read_records(path):
            if record["dataset"] in excluded_unseen:
                continue
            if record["task_id"] in unseen_seen_ids:
                raise ValueError(f"duplicate unseen evaluation task_id: {record['task_id']}")
            unseen_seen_ids.add(record["task_id"])
            unseen_records.append(record)
    if args.unseen_eval_every and not holdout_eval_records and not unseen_records:
        raise ValueError(
            "--unseen-eval-every requires --eval-pairs-per-task or retained records from --unseen-eval-input"
        )
    unseen_records = deterministic_fraction_per_dataset(
        unseen_records, args.unseen_eval_fraction, args.seed + 3
    ) if unseen_records else []
    unseen_rows = directions(unseen_records)
    periodic_eval_rows = holdout_eval_rows if holdout_eval_rows else unseen_rows
    periodic_eval_kind = "training_task_holdout" if holdout_eval_rows else "external_unseen_tasks"
    unseen_max_tokens = args.unseen_eval_max_new_tokens or (
        4096 if args.unseen_eval_mode == "reasoning" else 32
    )

    model_id = base.MODEL_IDS[args.model]
    model_path = (
        args.model_path.resolve()
        if args.model_path is not None
        else Path(snapshot_download(model_id, local_files_only=not args.allow_download))
    )
    if not model_path.is_dir():
        raise FileNotFoundError(f"model checkpoint directory not found: {model_path}")
    base.validate_model_files(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{local_rank}"}, attn_implementation="sdpa",
    )
    freeze_model(model)
    base.assert_default_rope(model)
    soft_prompt = initialize_soft_prompt(model, args.soft_tokens)
    if rank == 0:
        print(f"[precompute] caching {len(train_rows)} training directions in batches of {args.train_batch_size}", flush=True)
    training_states = precompute_training_states(
        model, tokenizer, train_rows, args.teacher_tokens, args.train_batch_size,
        show_progress=rank == 0,
    )
    optimizer = torch.optim.AdamW(
        [soft_prompt], lr=args.learning_rate, weight_decay=args.weight_decay
    )
    config = {
        "experiment": "universal_soft_suffix_sequence_kl",
        "model": args.model,
        "model_id": model_id,
        "inputs": inputs,
        "included_datasets": sorted(set(args.include_dataset)) or "all",
        "records": len(records),
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "train_directions": len(train_rows),
        "dev_directions": len(dev_rows),
        "task_families": {family: len(rows) for family, rows in sorted(train_pools.items())},
        "sampling": "uniform task-family round robin across ranks",
        "soft_tokens": args.soft_tokens,
        "hidden_size": model.config.hidden_size,
        "trainable_parameters": soft_prompt.numel(),
        "world_size": world_size,
        "directions_per_step": world_size,
        "train_batch_size": args.train_batch_size,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "teacher_tokens": args.teacher_tokens,
        "grad_clip": args.grad_clip,
        "offset_threshold": args.offset_threshold,
        "offset_scale": args.offset_scale,
        "anchor_weight": args.anchor_weight,
        "training_prompt_modes": training_prompt_modes,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "train_records_per_dataset": args.train_records_per_dataset or None,
        "unseen_evaluation": {
            "inputs": [str(path.resolve()) for path in args.unseen_eval_input],
            "excluded_datasets": sorted(excluded_unseen),
            "records": len(periodic_eval_rows) // 2,
            "directions": len(periodic_eval_rows),
            "kind": periodic_eval_kind,
            "pairs_per_training_task": args.eval_pairs_per_task or None,
            "fraction_per_dataset": args.unseen_eval_fraction,
            "every_steps": args.unseen_eval_every,
            "start_step": args.unseen_eval_start_step,
            "patience_steps": args.unseen_eval_patience,
            "mode": args.unseen_eval_mode,
            "max_new_tokens": unseen_max_tokens,
            "batch_size": args.unseen_eval_batch_size,
            "selection_metric": "soft_accuracy",
            "evaluation_impl": "batched_soft_only_generation",
            "kind": args.periodic_eval_kind,
        },
        "profile_step_timing": args.profile_step_timing,
        "initialization": "exact zero [soft_tokens, hidden_size]",
        "objective": "selective residual KL: repair Full/Direct offsets, preserve Direct otherwise, anchor to zero",
        "reasoning": args.explicit_reasoning,
        "boxed_output": True,
        "base_model_frozen": True,
        "rope_correction": "inverse_source_then_apply_target_using_model.rotary_emb",
        "torch_version": torch.__version__,
    }
    split_manifest = {
        "split": split_description,
        "seed": args.seed,
        "train_task_ids": [record["task_id"] for record in train_records],
        "dev_task_ids": [record["task_id"] for record in dev_records],
        "train_groups": sorted({group_key(record) for record in train_records}),
        "dev_groups": sorted({group_key(record) for record in dev_records}),
    }
    checkpoint_path = args.output / "best_latent_tokens.pt"
    history: list[dict[str, Any]] = []
    start_step = 1
    best_unseen_accuracy = float("-inf")
    best_unseen_step: int | None = None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        old_config = checkpoint["config"]
        comparable = {key: value for key, value in config.items() if key != "steps"}
        old_comparable = {key: old_config.get(key) for key in comparable}
        if old_comparable != comparable:
            raise ValueError("resume configuration differs from checkpoint")
        if args.steps <= int(checkpoint["step"]):
            raise ValueError("--steps must exceed the checkpoint step when resuming")
        soft_prompt.data.copy_(checkpoint["soft_prompt"].to(soft_prompt.device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint["history"])
        start_step = int(checkpoint["step"]) + 1
        for old_row in history:
            unseen = old_row.get("unseen_eval")
            if isinstance(unseen, dict):
                accuracy = unseen.get("soft_accuracy")
                if isinstance(accuracy, (int, float)) and accuracy > best_unseen_accuracy:
                    best_unseen_accuracy = float(accuracy)
                    best_unseen_step = int(old_row["step"])
        if rank == 0:
            base.write_json_atomic(args.output / "config.json", config)
    elif rank == 0:
        base.write_json_atomic(args.output / "config.json", config)
        base.write_json_atomic(args.output / "split_manifest.json", split_manifest)

    eval_rows = dev_rows[:args.eval_directions]
    if eval_rows and rank == 0 and not args.resume:
        initial_eval = evaluate_kl(
            model, tokenizer, soft_prompt, eval_rows, args.temperature, args.teacher_tokens,
            args.offset_threshold, args.offset_scale, args.explicit_reasoning,
        )
        base.write_json_atomic(args.output / "initial_eval.json", initial_eval)
        print(json.dumps({"step": 0, "dev": initial_eval}, ensure_ascii=False), flush=True)
    if distributed:
        dist.barrier()
    started = time.perf_counter()
    steps = range(start_step, args.steps + 1)
    progress = tqdm(steps, desc="train", unit="step", dynamic_ncols=True) if rank == 0 else steps
    for step in progress:
        batch = balanced_batch(train_pools, step, args.train_batch_size)
        record, target, row_reasoning = batch[0]
        timings_ms: dict[str, float] | None = {} if args.profile_step_timing else None
        static_rows = [training_states[(item[0]["task_id"], item[1], item[2])] for item in batch]
        teachers = [item[0] for item in static_rows]
        directs = [item[1] for item in static_rows]
        soft_rows = [(item[2], item[3], item[4]) for item in static_rows]
        optimizer.zero_grad(set_to_none=True)
        if timings_ms is not None:
            torch.cuda.synchronize(base.input_device(model))
            stage_started = time.perf_counter()
        students = continuation_logits_batch(model, soft_rows, soft_prompt)
        if timings_ms is not None:
            torch.cuda.synchronize(base.input_device(model))
            timings_ms["soft_forward"] = (time.perf_counter() - stage_started) * 1000
        per_row = [selective_residual_kl(
            teacher, direct, student, temperature=args.temperature,
            offset_threshold=args.offset_threshold * answer.numel(),
            offset_scale=args.offset_scale * answer.numel(),
        ) for teacher, direct, student, (_, _, answer) in zip(teachers, directs, students, soft_rows)]
        loss = torch.stack([value[0] / answer.numel() for value, (_, _, answer) in zip(per_row, soft_rows)]).mean()
        anchor = soft_prompt.square().mean()
        loss = loss + args.anchor_weight * anchor
        if timings_ms is not None:
            torch.cuda.synchronize(base.input_device(model))
            stage_started = time.perf_counter()
        loss.backward()
        if timings_ms is not None:
            torch.cuda.synchronize(base.input_device(model))
            timings_ms["backward"] = (time.perf_counter() - stage_started) * 1000
        if distributed:
            dist.all_reduce(soft_prompt.grad, op=dist.ReduceOp.SUM)
            soft_prompt.grad.div_(world_size)
        mean_loss = loss.detach().clone()
        if distributed:
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss.div_(world_size)
        grad_norm = float(torch.nn.utils.clip_grad_norm_([soft_prompt], args.grad_clip))
        if not torch.isfinite(loss) or not torch.isfinite(torch.tensor(grad_norm)):
            raise FloatingPointError(f"non-finite optimization value at step {step}")
        if timings_ms is not None:
            torch.cuda.synchronize(base.input_device(model))
            stage_started = time.perf_counter()
        optimizer.step()
        if timings_ms is not None:
            torch.cuda.synchronize(base.input_device(model))
            timings_ms["optimizer"] = (time.perf_counter() - stage_started) * 1000
        row: dict[str, Any] = {
            "step": step, "train_loss": float(mean_loss), "grad_norm": grad_norm,
            "rank0_task_id": record["task_id"], "rank0_dataset": record["dataset"],
            "rank0_target": target, "rank0_reasoning": batch[0][2],
            "answer_tokens_rank0": int(soft_rows[0][2].numel()),
            "rank0_task_family": task_family(record),
            "batch_directions": len(batch),
            "rank0_repair_weight": float(per_row[0][1].detach()),
            "rank0_full_direct_kl": float(per_row[0][2].detach() / soft_rows[0][2].numel()),
            "rank0_direct_soft_kl": float(per_row[0][3].detach() / soft_rows[0][2].numel()),
            "anchor": float(anchor.detach()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if timings_ms is not None:
            row["stage_timings_ms"] = timings_ms
        evaluate_now = eval_rows and (step == 1 or step % args.eval_every == 0 or step == args.steps)
        unseen_evaluate_now = (
            bool(periodic_eval_rows)
            and args.unseen_eval_every > 0
            and step >= args.unseen_eval_start_step
            and (step % args.unseen_eval_every == 0 or step == args.steps)
        )
        synchronize_evaluation = bool(evaluate_now or unseen_evaluate_now)
        if distributed and synchronize_evaluation:
            dist.barrier()
        if rank == 0 and evaluate_now:
            row["dev"] = evaluate_kl(
                model, tokenizer, soft_prompt, eval_rows, args.temperature,
                args.teacher_tokens, args.offset_threshold, args.offset_scale,
                args.explicit_reasoning,
            )
        stop_now = False
        if rank == 0 and unseen_evaluate_now:
            if args.periodic_eval_kind == "kl":
                kl_summary = evaluate_kl(
                    model, tokenizer, soft_prompt, periodic_eval_rows, args.temperature,
                    args.teacher_tokens, args.offset_threshold, args.offset_scale,
                    args.explicit_reasoning,
                )
                metric = -float(kl_summary["soft_kl"])
                row["unseen_eval"] = {
                    "kind": "teacher_forced_kl",
                    "selection_metric": "negative_soft_kl",
                    "selection_value": metric,
                    **kl_summary,
                }
            else:
                generation_summary = evaluate_soft_generation_batch(
                    model, tokenizer, soft_prompt, periodic_eval_rows, unseen_max_tokens,
                    args.unseen_eval_batch_size,
                    explicit_reasoning=args.unseen_eval_mode == "reasoning",
                    profile_stages=args.profile_step_timing,
                )
                metric = float(generation_summary["soft_accuracy"])
                row["unseen_eval"] = {
                    "kind": "generation",
                    "selection_metric": "soft_accuracy",
                    "selection_value": metric,
                    **generation_summary,
                }
            if metric > best_unseen_accuracy:
                best_unseen_accuracy = metric
                best_unseen_step = step
                save_checkpoint(checkpoint_path, soft_prompt, optimizer,
                                step, config, history + [row])
            elif (args.unseen_eval_patience and best_unseen_step is not None
                  and step - best_unseen_step >= args.unseen_eval_patience):
                stop_now = True
                row["early_stop"] = {
                    "reason": "unseen_soft_accuracy_no_improvement",
                    "best_step": best_unseen_step,
                    "best_selection_value": best_unseen_accuracy,
                    "patience_steps": args.unseen_eval_patience,
                }
        if distributed and synchronize_evaluation:
            dist.barrier()
            stop_signal = torch.tensor(int(stop_now), device=soft_prompt.device)
            dist.broadcast(stop_signal, src=0)
            stop_now = bool(stop_signal.item())
        if rank == 0:
            history.append(row)
            base.append_jsonl(args.output / "history.jsonl", row)
            if timings_ms is not None:
                base.append_jsonl(args.output / "step_timings.jsonl", {
                    "step": step,
                    "task_id": record["task_id"],
                    "target": target,
                    "stage_timings_ms": timings_ms,
                })
            print(json.dumps(row, ensure_ascii=False), flush=True)
            progress.set_postfix(
                loss=f"{float(mean_loss):.4f}",
                best=(f"{best_unseen_accuracy:.4f}" if best_unseen_step is not None else "--"),
            )
        if stop_now:
            break

    if rank != 0:
        dist.destroy_process_group()
        return 0
    final: dict[str, Any] = {
        "complete": True,
        "config": config,
        "last_step": history[-1],
        "best_unseen_step": best_unseen_step,
        "best_unseen_soft_accuracy": best_unseen_accuracy if best_unseen_step is not None else None,
        "best_latent_tokens": str(checkpoint_path) if best_unseen_step is not None else None,
        "wall_seconds": time.perf_counter() - started,
    }
    if args.save_final:
        final_checkpoint_path = args.output / "final_latent_tokens.pt"
        save_checkpoint(final_checkpoint_path, soft_prompt, optimizer, history[-1]["step"], config, history)
        final["final_latent_tokens"] = str(final_checkpoint_path)
    if args.generate_directions:
        generation_rows = [(record, target) for record, target, _ in dev_rows[:args.generate_directions]]
        final["generation"] = evaluate_generation(
            model, tokenizer, soft_prompt, generation_rows, args.max_new_tokens,
            explicit_reasoning=training_prompt_modes == ["reasoning"],
        )
        base.write_jsonl_atomic(
            args.output / "generation_samples.jsonl", final["generation"].pop("results")
        )
    base.write_json_atomic(args.output / "summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
