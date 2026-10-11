#!/usr/bin/env python3
"""Train a prefix-conditioned latent adapter to imitate Repeat-prefix repair."""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kv_semantic_attack.prefix_adapter_repeat import PrefixLatentAdapter, target_prefix_states_batch  # noqa: E402
from kv_semantic_attack.train_universal_soft_suffix import fixed_train_records, read_records, task_family  # noqa: E402
from kv_semantic_attack.universal_soft_suffix import (  # noqa: E402
    continuation_logits_batch,
    freeze_model,
    suffix_logits_batch,
)
from scripts import batch_eval as batch_ops  # noqa: E402
from scripts import run_direct_reuse as base  # noqa: E402
from kv_semantic_attack.teacher_trace_cache import cache_protocol, load_cache, save_cache
from kv_semantic_attack.frozen_prefix_cache import FrozenPrefixCache, relocate_block_shared_rope
from kv_semantic_attack.paired_full_context import PairedFullContext
from kv_semantic_attack import static_adapter_states
from kv_semantic_attack.full_teacher_forward import full_teacher_logits_batch
from kv_semantic_attack.tensor_batch import BatchedPast
from kv_semantic_attack.prepared_paired_tokens import PreparedPairedTokens
from kv_semantic_attack.paired_tensor_training import backward_paired_batch, paired_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--include-dataset", action="append", default=[],
                        help="restrict training to named datasets; omit to use every dataset in --input")
    parser.add_argument("--directions-file", type=Path,
                        help="JSONL rows with task_id and target (a or b); train only these directions")
    parser.add_argument("--model", choices=tuple(base.MODEL_IDS), default="4b")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-adapter-checkpoint", type=Path,
                        help="initialize adapter weights from an existing compatible checkpoint")
    parser.add_argument("--train-records-per-dataset", type=int, default=10)
    parser.add_argument("--use-all-records", action="store_true",
                        help="train from every record in --input after dataset filtering")
    parser.add_argument("--teacher-tokens", type=int, default=64)
    parser.add_argument('--teacher-precompute-batch-size',type=int,default=0,
                        help='generation-only batch; zero uses train-batch-size, effective training batch unchanged')
    parser.add_argument('--audit-task-loss',action='store_true',help='record per-task CE/KL and coverage each epoch')
    parser.add_argument("--teacher-trace-cache", type=Path,
                        help="pregenerate all mixed Full reasoning token traces once before training")
    parser.add_argument('--frozen-prefix-cache-gib', type=float, default=0,
                        help='bounded prefix KV/state GPU cache; zero keeps reference path')
    parser.add_argument('--paired-full-context-gib', type=float, default=0,
                        help='combine teacher/context forward with bounded reusable role contexts')
    parser.add_argument('--paired-role-forward', action='store_true',
                        help='pair A/B and prompt modes per batch; one dense frozen forward, no dataset KV cache')
    parser.add_argument('--static-training-states', action='store_true',
                        help='precompute teacher logits/reused KV/prefix states once; student-only epochs')
    parser.add_argument('--single-full-teacher-forward', action='store_true',
                        help='mixed cached traces: one teacher forward only on reasoning rows')
    parser.add_argument("--no-reasoning-teacher-tokens", type=int, default=4,
                        help="maximum generated tokens for direct boxed-answer teachers")
    parser.add_argument("--teacher-mode", choices=("repeat", "full"), default="repeat",
                        help="teacher context: reused KV plus repeated prefix, or dense target full prefill")
    parser.add_argument("--teacher-trace", choices=("generated", "gold", "mixed"), default="generated",
                        help="generated, gold, or mixed: no-reasoning gold first-token CE and reasoning Full generated KD")
    parser.add_argument("--mixed-gold-full", action="store_true",
                        help="no-reasoning: CE on complete gold continuation; reasoning: Full generated-trace KL")
    parser.add_argument("--latent-tokens", type=int, default=32)
    parser.add_argument("--adapter-bottleneck", type=int, default=256)
    parser.add_argument("--prefix-pooling", choices=("last", "mean"), default="last",
                        help="pooling used to encode the frozen target prefix")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=0,
                        help="complete epochs to train; overrides --steps when positive")
    parser.add_argument("--complete-pool-epochs", action="store_true", default=None,
                        help="visit every direction once per epoch; default when --epochs > 0")
    parser.add_argument("--legacy-family-cycle", action="store_false", dest="complete_pool_epochs",
                        help="explicitly reproduce old incomplete family-cycle sampling")
    parser.add_argument("--save-epoch-checkpoints", action="store_true",
                        help="save every epoch for external validation-based selection")
    parser.add_argument("--train-batch-size", type=int, default=1,
                        help="directions per optimizer update")
    parser.add_argument("--train-reasoning", choices=("no", "yes", "both"), default="yes",
                        help="prompt/generation protocols included in training")
    parser.add_argument("--latest-save-interval", type=int, default=100,
                        help="save latest_prefix_adapter.pt every N optimizer updates; 0 saves only at exit")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--suffix-kl-weight", type=float, default=0.0,
                        help="additional KL weight over all target suffix tokens after the reused block")
    parser.add_argument("--gold-first-token-ce-weight", type=float, default=0.0,
                        help="cross-entropy weight on the first gold answer token")
    parser.add_argument("--gold-first-token-ce-vocab", choices=("full", "options"), default="full",
                        help="compute gold CE over the full vocabulary or only legal option letters")
    parser.add_argument("--kd-weight", type=float, default=1.0,
                        help="answer-distribution KD weight")
    parser.add_argument("--kd-directions-file", type=Path,
                        help="JSONL task_id/target whitelist for KD; gold CE still uses all selected directions")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-wall-seconds", type=float, default=0.0,
                        help="save a checkpoint and stop cleanly after this training time; 0 disables")
    parser.add_argument("--early-stopping-patience-epochs", type=int, default=0,
                        help="stop after this many complete epochs without a lower mean loss; 0 disables")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                        help="required decrease in epoch mean loss to reset early stopping")
    parser.add_argument("--adaptive-microbatch", action='store_true',
                        help='paired-role only: retry OOM with smaller complete-pair microbatches; preserve effective batch')
    args = parser.parse_args()
    if args.audit_task_loss and not args.paired_role_forward:
        parser.error('audit-task-loss requires paired-role-forward')
    if args.teacher_precompute_batch_size < 0:
        parser.error('teacher-precompute-batch-size must be nonnegative')
    if args.adaptive_microbatch and (not args.paired_role_forward or args.kd_directions_file
                                    or args.gold_first_token_ce_vocab != 'full'):
        parser.error('adaptive-microbatch requires paired-role, full-vocabulary CE and no KD whitelist')
    if args.complete_pool_epochs is None:
        args.complete_pool_epochs = args.epochs > 0
    if args.teacher_trace_cache and args.teacher_trace != 'mixed':
        parser.error('teacher-trace-cache requires mixed supervision')
    if args.paired_role_forward and (args.teacher_trace != 'mixed' or not args.teacher_trace_cache
            or not args.complete_pool_epochs or args.train_batch_size % 4 != 0
            or args.static_training_states or args.suffix_kl_weight or args.mixed_gold_full
            or args.frozen_prefix_cache_gib or args.paired_full_context_gib or args.single_full_teacher_forward):
        parser.error('paired-role-forward requires cached mixed, complete epochs, batch multiple of4, no other optimization modes')
    if args.static_training_states and (args.teacher_trace != 'mixed' or not args.teacher_trace_cache
            or args.suffix_kl_weight or args.mixed_gold_full or args.paired_full_context_gib
            or args.frozen_prefix_cache_gib or args.single_full_teacher_forward):
        parser.error('static states requires cached mixed reference path without experimental optimizations')
    if args.paired_full_context_gib > 0 and (args.teacher_trace != 'mixed' or
            not args.teacher_trace_cache or args.suffix_kl_weight or args.mixed_gold_full):
        parser.error('paired context requires cached mixed supervision with suffix KL disabled')
    if args.single_full_teacher_forward and (not args.teacher_trace_cache or args.teacher_trace != 'mixed'):
        parser.error('single-full-teacher-forward requires mixed and teacher-trace-cache')
    if min(args.train_records_per_dataset, args.teacher_tokens, args.no_reasoning_teacher_tokens, args.latent_tokens, args.adapter_bottleneck, args.steps, args.train_batch_size) < 1:
        parser.error("counts must be positive")
    if args.epochs < 0:
        parser.error("epochs must be non-negative")
    if args.complete_pool_epochs and not args.epochs:
        parser.error("complete-pool-epochs requires --epochs > 0")
    if args.latest_save_interval < 0:
        parser.error("latest-save-interval must be non-negative")
    if args.max_wall_seconds < 0:
        parser.error("max-wall-seconds must be non-negative")
    if args.early_stopping_patience_epochs < 0 or args.early_stopping_min_delta < 0:
        parser.error("early stopping values must be non-negative")
    if min(args.suffix_kl_weight, args.gold_first_token_ce_weight, args.kd_weight) < 0:
        parser.error("loss weights must be non-negative")
    if args.suffix_kl_weight and args.teacher_mode != "full":
        parser.error("suffix KL currently requires --teacher-mode full")
    if args.teacher_trace == "gold" and args.train_reasoning != "no":
        parser.error("gold teacher traces currently support only --train-reasoning no")
    if args.gold_first_token_ce_weight and args.teacher_trace not in {"gold", "mixed"}:
        parser.error("gold first-token CE requires --teacher-trace gold or mixed")
    if args.teacher_trace == "mixed" and (
        args.teacher_mode != "full" or args.train_reasoning != "both" or
        args.kd_weight != 1.0 or args.gold_first_token_ce_weight != 1.0 or
        args.gold_first_token_ce_vocab != "full" or args.suffix_kl_weight != 0.0 or
        args.teacher_tokens > 64 or args.kd_directions_file or args.init_adapter_checkpoint or
        args.mixed_gold_full
    ):
        parser.error("mixed requires full/both, kd-weight=1, gold-first-token-ce-weight=1, "
                     "full CE vocabulary, suffix-kl-weight=0, teacher-tokens<=64, "
                     "from-scratch initialization, no KD filters or mixed-gold-full")
    if args.kd_directions_file and not args.kd_weight:
        parser.error("--kd-directions-file requires a positive --kd-weight")
    if args.mixed_gold_full and (
        args.teacher_mode != "full" or args.teacher_trace != "generated" or
        args.train_reasoning != "both" or args.gold_first_token_ce_weight or
        args.suffix_kl_weight or args.kd_directions_file or args.kd_weight != 1.0
    ):
        parser.error("mixed-gold-full requires full/generated/both, kd-weight=1, no other loss flags or KD filters")
    return args


def parts(tokenizer: Any, record: dict[str, Any], side: str, reasoning: bool) -> base.PromptParts:
    # Both prompt variants disable native thinking; only the instruction differs.
    return base.build_prompt_parts(tokenizer, record, side, enable_thinking=False, explicit_reasoning=reasoning, boxed_output=True)


@torch.no_grad()
def reuse_pasts_batch(
    model: Any, targets: list[base.PromptParts], sources: list[base.PromptParts],
    prefix_layers=None, shared_rope=False,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    for target, source in zip(targets, sources):
        if not torch.equal(target.block_ids, source.block_ids):
            raise ValueError("source and target block token IDs differ")
    _, source_layers = batch_ops.prefill_ids_batch(model, [part.full_ids for part in sources])
    if prefix_layers is None:
        _, prefix_layers = batch_ops.prefill_ids_batch(model, [part.prefix_ids for part in targets])
    mixed = []
    for target, source, donor_cache, prefix_cache in zip(targets, sources, source_layers, prefix_layers):
        donor = base.slice_cache(donor_cache, source.block_token_start, source.block_token_end)
        relocate = relocate_block_shared_rope if shared_rope else base.relocate_block
        donor = relocate(model, donor, source.block_token_start, target.block_token_start)
        mixed.append(base.splice_prefix_block(prefix_cache, donor))
    return mixed


@torch.no_grad()
def generate_teachers_batch(
    model: Any, tokenizer: Any, pasts: list[list[tuple[torch.Tensor, torch.Tensor]]],
    suffixes: list[torch.Tensor], tokens: int, stop_at_closing_brace: list[bool],
    no_reasoning_tokens: int,
) -> list[torch.Tensor]:
    traces: list[list[int] | None] = [None] * len(pasts)
    for stop in (False, True):
        indices = [index for index, value in enumerate(stop_at_closing_brace) if value == stop]
        if not indices:
            continue
        logits, caches = batch_ops.forward_suffix_batch(
            model, [pasts[index] for index in indices], [suffixes[index] for index in indices],
        )
        lengths = [int(cache[0][0].shape[-2]) for cache in caches]
        padded = batch_ops.pad_cache(list(zip(caches, lengths)), max(lengths))
        decoded = batch_ops.decode_batch(
            model, tokenizer, logits, padded, lengths,
            no_reasoning_tokens if stop else tokens,
            stop_at_closing_brace=[stop] * len(indices),
        )
        for index, trace in zip(indices, decoded):
            traces[index] = trace
    return [torch.tensor(trace, dtype=torch.long, device=base.input_device(model)) for trace in traces if trace is not None]


@torch.no_grad()
def full_teacher_pasts_batch(
    model: Any, targets: list[base.PromptParts],
) -> tuple[list[list[tuple[torch.Tensor, torch.Tensor]]], list[torch.Tensor]]:
    """Return dense target caches split before their final prompt token."""
    _, full_layers = batch_ops.prefill_ids_batch(model, [target.full_ids for target in targets])
    pasts=[]; suffixes=[]
    for target, layers in zip(targets, full_layers):
        length=int(target.full_ids.numel())
        pasts.append(base.slice_cache(layers, 0, length - 1))
        suffixes.append(target.full_ids[-1:].clone())
    return pasts, suffixes


@torch.no_grad()
def full_teacher_contexts_batch(
    model: Any, targets: list[base.PromptParts], include_suffix: bool,
) -> tuple[
    list[list[tuple[torch.Tensor, torch.Tensor]]], list[torch.Tensor],
    list[list[tuple[torch.Tensor, torch.Tensor]]] | None, list[torch.Tensor] | None,
]:
    """Build answer contexts and optionally contexts that predict the entire suffix."""
    _, full_layers = batch_ops.prefill_ids_batch(model, [target.full_ids for target in targets])
    answer_pasts = []
    answer_suffixes = []
    suffix_pasts = [] if include_suffix else None
    suffix_seeds = [] if include_suffix else None
    for target, layers in zip(targets, full_layers):
        length = int(target.full_ids.numel())
        answer_pasts.append(base.slice_cache(layers, 0, length - 1))
        answer_suffixes.append(target.full_ids[-1:].clone())
        if include_suffix:
            seed_index = target.block_token_end - 1
            if seed_index < 0:
                raise ValueError("cannot distill suffix without a preceding block token")
            assert suffix_pasts is not None and suffix_seeds is not None
            suffix_pasts.append(base.slice_cache(layers, 0, seed_index))
            suffix_seeds.append(target.full_ids[seed_index:seed_index + 1].clone())
    return answer_pasts, answer_suffixes, suffix_pasts, suffix_seeds


def gold_traces_batch(
    tokenizer: Any, batch: list[tuple[dict[str, Any], str, bool]], device: torch.device,
) -> list[torch.Tensor]:
    """Return exact continuations after the no-reasoning ``\\boxed{`` prefill."""
    traces = []
    for record, side, reasoning in batch:
        if reasoning:
            raise ValueError("gold boxed-answer traces are defined only for no-reasoning prompts")
        token_ids = tokenizer.encode(f"{record[f'gold_{side}']}}}", add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"empty gold trace for {record['task_id']} target {side}")
        traces.append(torch.tensor(token_ids, dtype=torch.long, device=device))
    return traces


def kl(teacher: torch.Tensor, student: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.nn.functional.kl_div(torch.log_softmax(student.float() / temperature, -1), torch.softmax(teacher.float() / temperature, -1), reduction="batchmean") * temperature ** 2


@torch.no_grad()
def mixed_traces_batch(
    model: Any, tokenizer: Any, batch: list[tuple[dict[str, Any], str, bool]],
    teacher_pasts: list, teacher_suffixes: list[torch.Tensor], teacher_tokens: int,
    trace_cache: dict | None = None,
) -> list[torch.Tensor]:
    """Keep batch order; only explicit-reasoning items use generated Full traces."""
    traces: list[torch.Tensor | None] = [None] * len(batch)
    direct = [i for i, (_, _, reasoning) in enumerate(batch) if not reasoning]
    reasoning = [i for i, (_, _, reasoning) in enumerate(batch) if reasoning]
    if direct:
        gold = gold_traces_batch(tokenizer, [batch[i] for i in direct], base.input_device(model))
        for i, trace in zip(direct, gold):
            traces[i] = trace
    if reasoning:
        if trace_cache is not None:
            generated = [torch.tensor(trace_cache[(batch[i][0]['task_id'], batch[i][1])],
                                      dtype=torch.long, device=base.input_device(model)) for i in reasoning]
        else:
            generated = generate_teachers_batch(
                model, tokenizer, [teacher_pasts[i] for i in reasoning],
                [teacher_suffixes[i] for i in reasoning], teacher_tokens,
                stop_at_closing_brace=[False] * len(reasoning),
                no_reasoning_tokens=teacher_tokens,
            )
        for i, trace in zip(reasoning, generated):
            traces[i] = trace
    if any(trace is None for trace in traces):
        raise ValueError("missing mixed teacher trace")
    return [trace for trace in traces if trace is not None]


def masked_answer_kd(
    teacher: list[torch.Tensor], student: list[torch.Tensor],
    indices: list[int], temperature: float,
) -> torch.Tensor:
    """Mean over selected examples (and trace tokens), never over masked examples."""
    if indices:
        lengths = [student[i].shape[0] for i in indices]
        reference = torch.cat([teacher[i].detach() for i in indices]).float()
        prediction = torch.cat([student[i] for i in indices]).float()
        if reference.shape != prediction.shape:
            raise ValueError("teacher/student answer spans differ")
        per_token = torch.nn.functional.kl_div(
            torch.log_softmax(prediction / temperature, -1),
            torch.softmax(reference / temperature, -1), reduction="none").sum(-1)
        counts = torch.tensor(lengths, device=prediction.device)
        weights = torch.repeat_interleave(counts.float().reciprocal(), counts)
        return (per_token * weights).sum() * (temperature ** 2 / len(indices))
    return torch.stack([prediction.reshape(-1)[0] for prediction in student]).sum() * 0.0


def masked_gold_first_token_ce(
    student: list[torch.Tensor], traces: list[torch.Tensor], indices: list[int],
) -> torch.Tensor:
    """The v2 full-vocabulary first-token objective, NOT gold sequence CE."""
    if indices:
        return torch.nn.functional.cross_entropy(
            torch.stack([student[i][0] for i in indices]).float(),
            torch.stack([traces[i][0] for i in indices]))
    return torch.stack([prediction.reshape(-1)[0] for prediction in student]).sum() * 0.0


@torch.no_grad()
def mixed_supervision(model, tokenizer, batch, targets, teacher_tokens):
    """Generate Full traces ONLY for reasoning; gold never depends on Full correctness."""
    traces = [None] * len(batch)
    teacher = [None] * len(batch)
    direct = [i for i, (_, _, reasoning) in enumerate(batch) if not reasoning]
    reasoning = [i for i, (_, _, reasoning) in enumerate(batch) if reasoning]
    if direct:
        gold = gold_traces_batch(tokenizer, [batch[i] for i in direct], base.input_device(model))
        for i, trace in zip(direct, gold):
            traces[i] = trace
    if reasoning:
        pasts, suffixes = full_teacher_pasts_batch(model, [targets[i] for i in reasoning])
        generated = generate_teachers_batch(model, tokenizer, pasts, suffixes,
                                           teacher_tokens, [False] * len(reasoning), teacher_tokens)
        logits = continuation_logits_batch(model, list(zip(pasts, suffixes, generated)), None)
        for i, trace, prediction in zip(reasoning, generated, logits):
            traces[i] = trace
            teacher[i] = prediction
    if any(trace is None for trace in traces):
        raise ValueError("missing mixed-supervision trace")
    return traces, teacher


def mixed_supervision_loss(student, teacher, traces, reasoning, temperature):
    """Equal per-example weighting; CE/KL each normalized over its trace tokens."""
    zero = student[0].new_zeros(())
    ce_terms, kd_terms = [], []
    if not (len(student) == len(teacher) == len(traces) == len(reasoning)):
        raise ValueError("mixed-supervision batch length mismatch")
    for prediction, reference, trace, is_reasoning in zip(student, teacher, traces, reasoning):
        if is_reasoning:
            if reference is None:
                raise ValueError("reasoning item requires Full teacher logits")
            kd_terms.append(kl(reference.detach(), prediction, temperature))
        else:
            if reference is not None:
                raise ValueError("no-reasoning must not receive Full teacher logits")
            ce_terms.append(torch.nn.functional.cross_entropy(prediction.float(), trace))
    ce = torch.stack(ce_terms).sum() / len(student) if ce_terms else zero
    kd = torch.stack(kd_terms).sum() / len(student) if kd_terms else zero
    return ce + kd, ce, kd


def load_direction_set(path: Path) -> set[tuple[str, str]]:
    directions: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        target = row.get("target")
        if target not in {"a", "b"}:
            raise ValueError(f"{path}: target must be a or b")
        direction = (row["task_id"], target)
        if direction in directions:
            raise ValueError(f"{path}: duplicate direction {direction}")
        directions.add(direction)
    if not directions:
        raise ValueError(f"{path}: no directions selected")
    return directions


@torch.no_grad()
def prepare_teacher_trace_cache(model, tokenizer, pools, args):
    started = time.perf_counter()
    directions = [(record, side, reason) for pool in pools.values()
                  for record, side, reason in pool if reason]
    directions.sort(key=lambda row: (row[0]['task_id'], row[1]))
    expected = {(record['task_id'], side) for record, side, _ in directions}
    records = sorted({record['task_id']: record for record, _, _ in directions}.values(),
                     key=lambda record: record['task_id'])
    protocol = cache_protocol(records, args.model_path, args.teacher_tokens)
    cached = load_cache(args.teacher_trace_cache, protocol, expected)
    loaded_count = len(cached)
    load_seconds = time.perf_counter() - started
    missing = [row for row in directions if (row[0]['task_id'], row[1]) not in cached]
    print(f'[teacher-cache] loaded={loaded_count} missing={len(missing)} '
          f'load_seconds={load_seconds:.4f}; generate only missing directions', flush=True)
    generation_batch = getattr(args,'teacher_precompute_batch_size',0) or args.train_batch_size
    for start in tqdm(range(0, len(missing), generation_batch), desc='Full teacher precompute', unit='batch'):
        batch = missing[start:start+generation_batch]
        targets = [parts(tokenizer, record, side, True) for record, side, _ in batch]
        pasts, suffixes = full_teacher_pasts_batch(model, targets)
        generated = generate_teachers_batch(model, tokenizer, pasts, suffixes, args.teacher_tokens,
                                            [False]*len(batch), args.teacher_tokens)
        if len(generated) != len(batch):
            raise ValueError('Full teacher precompute count mismatch')
        for (record, side, _), trace in zip(batch, generated):
            cached[(record['task_id'], side)] = trace.tolist()
        save_cache(args.teacher_trace_cache, protocol, cached, expected)
    if set(cached) != expected:
        raise ValueError('missing precomputed teacher directions')
    elapsed = time.perf_counter() - started
    base.write_json_atomic(args.output / 'teacher_cache_metrics.json', {
        'cache_path':str(args.teacher_trace_cache), 'loaded_directions':loaded_count,
        'generated_directions':len(missing), 'total_directions':len(cached),
        'load_seconds':load_seconds, 'total_seconds':elapsed,
        'generation_and_save_seconds':elapsed-load_seconds if missing else 0.0,
        'cache_hit':not missing,
        'generation_batch_size':generation_batch,
    })
    print(f'[teacher-cache] complete {len(cached)} directions; no generate inside epochs', flush=True)
    return cached


def paired_role_epoch_batches(pools, seed, batch_size=4):
    """Pack whole pairs across families; never split or repeat a direction."""
    if batch_size < 4 or batch_size % 4:
        raise ValueError('paired batch size must be a positive multiple of4')
    rng = random.Random(seed)
    per_family = {}
    for family in sorted(pools):
        grouped = defaultdict(list)
        for item in pools[family]:
            grouped[item[0]['task_id']].append(item)
        chunks = []
        for task in sorted(grouped):
            chunk = sorted(grouped[task], key=lambda row: (row[1],row[2]))
            if len(chunk) != 4 or {(s,m) for _,s,m in chunk} != {('a',False),('a',True),('b',False),('b',True)}:
                raise ValueError('paired role training requires both roles and both modes per pair')
            chunks.append(chunk)
        rng.shuffle(chunks); per_family[family] = chunks
    ordered = [(family,chunks[i]) for i in range(max(map(len,per_family.values())))
               for family,chunks in per_family.items() if i < len(chunks)]
    per_update = batch_size // 4
    return [(items[0][0] if len(items)==1 else 'mixed_paired_families',
             [row for _,chunk in items for row in chunk])
            for start in range(0,len(ordered),per_update)
            for items in [ordered[start:start+per_update]]]


def complete_epoch_batches(pools, batch_size, seed):
    """Visit every direction once, without wrapping smaller task pools."""
    rng = random.Random(seed)
    per_family = {}
    for family in sorted(pools):
        shuffled = list(pools[family])
        rng.shuffle(shuffled)
        per_family[family] = [shuffled[i:i+batch_size]
                              for i in range(0, len(shuffled), batch_size)]
    batches = []
    for index in range(max(map(len, per_family.values()))):
        for family, chunks in per_family.items():
            if index < len(chunks):
                batches.append((family, chunks[index]))
    return batches


def main() -> int:
    args = parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    if args.output.exists() and any(args.output.iterdir()): raise ValueError(f"nonempty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    records=[]; seen=set(); allowed=set(args.include_dataset)
    for path in args.input:
        for record in read_records(path):
            if (not allowed or record["dataset"] in allowed) and record["task_id"] not in seen:
                records.append(record); seen.add(record["task_id"])
    if not allowed:
        allowed = {r["dataset"] for r in records}
    if {r["dataset"] for r in records} != allowed: raise ValueError("selected datasets absent from inputs")
    if args.use_all_records:
        train = records
    else:
        train, _ = fixed_train_records(records, args.train_records_per_dataset, args.seed)
    modes = (False, True) if args.train_reasoning == "both" else (args.train_reasoning == "yes",)
    selected_directions: set[tuple[str, str]] | None = None
    if args.directions_file:
        selected_directions = load_direction_set(args.directions_file)
        available = {(record["task_id"], side) for record in train for side in ("a", "b")}
        missing = selected_directions - available
        if missing:
            raise ValueError(f"{args.directions_file}: {len(missing)} requested directions missing from training inputs")
        if not selected_directions:
            raise ValueError(f"{args.directions_file}: no directions selected")
    kd_directions = load_direction_set(args.kd_directions_file) if args.kd_directions_file else None
    if kd_directions is not None:
        available = {(record["task_id"], side) for record in train for side in ("a", "b")}
        missing = kd_directions - available
        if missing:
            raise ValueError(f"{args.kd_directions_file}: {len(missing)} KD directions missing from training inputs")
    pools: dict[str, list[tuple[dict[str, Any], str, bool]]] = defaultdict(list)
    for record in train:
        for side in ("a", "b"):
            if selected_directions is not None and (record["task_id"], side) not in selected_directions:
                continue
            for reasoning in modes:
                pools[task_family(record)].append((record, side, reasoning))
    for index, name in enumerate(sorted(pools)): random.Random(args.seed + index).shuffle(pools[name])
    families=sorted(pools)
    base.validate_model_files(args.model_path)
    tokenizer=AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    freeze_model(model); base.assert_default_rope(model)
    trace_cache = (prepare_teacher_trace_cache(model, tokenizer, pools, args)
                   if args.teacher_trace_cache else None)
    prepared_prompts, prepared_traces = {}, {}
    if args.paired_role_forward:
        # Tokenize once; keep only token tensors/metadata, never dataset-wide KV.
        for pool in pools.values():
            for record, side, reasoning in pool:
                entry = (record['task_id'], side, reasoning)
                prepared_prompts[entry] = parts(tokenizer,record,side,reasoning)
                prepared_traces[entry] = (
                    torch.tensor(trace_cache[(record['task_id'],side)],dtype=torch.long,
                                 device=base.input_device(model)) if reasoning else
                    gold_traces_batch(tokenizer,[(record,side,reasoning)],base.input_device(model))[0])
    prepared_token_store = (PreparedPairedTokens(prepared_prompts,prepared_traces,base.input_device(model))
                            if args.paired_role_forward else None)
    frozen_training_states = (static_adapter_states.precompute(
        sys.modules[__name__],model,tokenizer,pools,args,trace_cache)
        if args.static_training_states else None)
    # Precomputation must not change adapter initialization or epoch RNG.
    random.seed(args.seed); torch.manual_seed(args.seed)
    adapter=PrefixLatentAdapter(model.config.hidden_size, args.latent_tokens, args.adapter_bottleneck).to(base.input_device(model))
    if args.init_adapter_checkpoint:
        initial = torch.load(args.init_adapter_checkpoint, map_location="cpu", weights_only=False)
        initial_config = initial.get("config", {})
        for key, expected in (("latent_tokens", args.latent_tokens), ("adapter_bottleneck", args.adapter_bottleneck)):
            if initial_config.get(key) != expected:
                raise ValueError(f"init checkpoint {key}={initial_config.get(key)!r}, expected {expected!r}")
        adapter.load_state_dict(initial["adapter"])
        print(f"[init] loaded adapter from {args.init_adapter_checkpoint}", flush=True)
    optimizer=torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
    steps_per_epoch = sum(math.ceil(len(pool) / args.train_batch_size) for pool in pools.values())
    if args.paired_role_forward:
        steps_per_epoch = len(paired_role_epoch_batches(pools,args.seed,args.train_batch_size))
    if args.epochs:
        args.steps = args.epochs * steps_per_epoch
    config={"experiment":"prefix_adapter_repeat_teacher","datasets":sorted(allowed),"teacher":args.teacher_mode,"teacher_trace":args.teacher_trace,"teacher_tokens":args.teacher_tokens,"no_reasoning_teacher_tokens":args.no_reasoning_teacher_tokens,"suffix_kl_weight":args.suffix_kl_weight,"gold_first_token_ce_weight":args.gold_first_token_ce_weight,"gold_first_token_ce_vocab":args.gold_first_token_ce_vocab,"kd_weight":args.kd_weight,"kd_directions_file":str(args.kd_directions_file) if args.kd_directions_file else None,"kd_directions":len(kd_directions) if kd_directions is not None else 2*len(train),"latent_tokens":args.latent_tokens,"adapter_bottleneck":args.adapter_bottleneck,"prefix_pooling":args.prefix_pooling,"steps":args.steps,"epochs_requested":args.epochs,"train_batch_size":args.train_batch_size,"train_reasoning":args.train_reasoning,"latest_save_interval":args.latest_save_interval,"use_all_records":args.use_all_records,"training_records":len(train),"init_adapter_checkpoint":str(args.init_adapter_checkpoint.resolve()) if args.init_adapter_checkpoint else None,"directions_file":str(args.directions_file) if args.directions_file else None,"selected_base_directions":len(selected_directions) if selected_directions is not None else 2*len(train),"directions_per_epoch":sum(len(pool) for pool in pools.values()),"trained_task_families":len(pools),"steps_per_epoch":steps_per_epoch,"max_wall_seconds":args.max_wall_seconds,"early_stopping_patience_epochs":args.early_stopping_patience_epochs,"early_stopping_min_delta":args.early_stopping_min_delta,"model":args.model,"rope_correction":"inverse_source_then_apply_target"}
    config["enable_thinking"] = False
    config['teacher_trace_cache'] = str(args.teacher_trace_cache) if args.teacher_trace_cache else None
    config['teacher_generation_schedule'] = 'once_before_training' if trace_cache is not None else 'online_per_step'
    config['teacher_precompute_batch_size'] = args.teacher_precompute_batch_size or args.train_batch_size
    config['frozen_prefix_cache_gib'] = args.frozen_prefix_cache_gib
    config['paired_full_context_gib'] = args.paired_full_context_gib
    config['paired_role_forward'] = args.paired_role_forward
    config['static_training_states'] = args.static_training_states
    config['single_full_teacher_forward'] = args.single_full_teacher_forward
    config["complete_pool_epochs"] = args.complete_pool_epochs
    config["sampling_protocol"] = ("each_direction_once_per_epoch_v1" if args.complete_pool_epochs
                                   else "legacy_uniform_family_cycle")
    config["save_epoch_checkpoints"] = args.save_epoch_checkpoints
    config["prompt_protocol"] = "explicit_prompt_thinking_off_v1"
    config["tensorized_paired_flow"] = bool(args.paired_role_forward)
    config['prepared_token_table_bytes'] = prepared_token_store.bytes if prepared_token_store is not None else 0
    config["adaptive_microbatch"] = args.adaptive_microbatch
    config['audit_task_loss']=args.audit_task_loss
    config["supervision_protocol"] = "no_reasoning_gold_sequence_ce__reasoning_full_kl_v1" if args.mixed_gold_full else "legacy"
    config["mixed_gold_full"] = args.mixed_gold_full
    if args.teacher_trace == "mixed":
        config["supervision_protocol"] = "no_reasoning_gold_first_token_ce__reasoning_full_generated_kd_v1"
        config["supervision_modes"] = {
            "no_reasoning": {"trace": "gold_side_plus_closing_brace", "loss": "gold_first_token_ce",
                             "vocabulary": "full", "weight": 1.0, "explicit_reasoning": False,
                             "enable_thinking": False},
            "reasoning": {"trace": "generated", "teacher": "full", "loss": "kd",
                          "max_new_tokens": args.teacher_tokens, "weight": 1.0,
                          "explicit_reasoning": True, "enable_thinking": False},
        }
        config["supervision_reduction"] = "mean_within_each_mode; KD mean_over_trace_tokens"
        config["initialization_protocol"] = "from_scratch"
    base.write_json_atomic(args.output / "config.json", config)
    started=time.perf_counter(); history=args.output / "history.jsonl"; progress=tqdm(range(1,args.steps+1),desc="repeat-adapter",unit="step")
    completed_steps = 0
    best_epoch_loss = float("inf")
    epochs_without_improvement = 0
    epoch_losses: list[float] = []
    epoch_batches = []
    prefix_cache = (FrozenPrefixCache(model,args.prefix_pooling,int(args.frozen_prefix_cache_gib*2**30))
                    if args.frozen_prefix_cache_gib > 0 else None)
    paired_context = (PairedFullContext(model,args.prefix_pooling,int(args.paired_full_context_gib*2**30))
                      if args.paired_full_context_gib > 0 else None)
    if args.paired_role_forward:
        paired_context = PairedFullContext(model,args.prefix_pooling,max_bytes=0)
    paired_microbatch = args.train_batch_size
    task_loss_totals = {}
    for step in progress:
        if args.max_wall_seconds and time.perf_counter() - started >= args.max_wall_seconds:
            print(f"[stop] reached max-wall-seconds={args.max_wall_seconds}", flush=True)
            break
        if args.complete_pool_epochs:
            epoch_offset = (step - 1) % steps_per_epoch
            if epoch_offset == 0:
                epoch_seed = args.seed + (step - 1) // steps_per_epoch
                epoch_batches = (paired_role_epoch_batches(pools,epoch_seed,args.train_batch_size) if args.paired_role_forward
                                 else complete_epoch_batches(pools,args.train_batch_size,epoch_seed))
            family, batch = epoch_batches[epoch_offset]
        else:
            family=families[(step-1)%len(families)]
            pool=pools[family]
            offset=((step-1)*args.train_batch_size//len(families)) % len(pool)
            batch=[pool[(offset + index) % len(pool)] for index in range(args.train_batch_size)]
        targets = sources = repeated = None
        if args.paired_role_forward:
            targets=[prepared_prompts[(record['task_id'],side,mode)] for record,side,mode in batch]
            sources=[prepared_prompts[(record['task_id'],'b' if side=='a' else 'a',mode)]
                     for record,side,mode in batch]
        elif frozen_training_states is None:
            targets=[parts(tokenizer,record,side,reasoning) for record,side,reasoning in batch]
            sources=[parts(tokenizer,record,"b" if side=="a" else "a",reasoning) for record,side,reasoning in batch]
            repeated=[parts(tokenizer,base.with_repeated_target_prefix(record,side),side,reasoning) for record,side,reasoning in batch]
        optimizer.zero_grad(set_to_none=True)
        if args.paired_role_forward and args.gold_first_token_ce_vocab == 'full':
            audit_rows=[] if args.audit_task_loss else None
            def compute_paired(chunk):
                return paired_loss(model,adapter,paired_context,chunk,prepared_prompts,
                                   prepared_traces,args,kd_directions,masked_answer_kd,
                                   masked_gold_first_token_ce,token_store=prepared_token_store,audit_rows=audit_rows)
            if args.adaptive_microbatch:
                values, actual_microbatch, oom_retries = backward_paired_batch(
                    batch,compute_paired,adapter.parameters(),paired_microbatch,optimizer=optimizer,
                    reset_audit=audit_rows.clear if audit_rows is not None else None)
                # A short epoch-tail batch must not shrink the persistent cap.
                if oom_retries:
                    paired_microbatch = actual_microbatch
            else:
                values = compute_paired(batch)
                values['loss'].backward()
                actual_microbatch, oom_retries = len(batch), 0
            loss, answer_loss, gold_ce_loss = (values[key] for key in ('loss','answer_loss','gold_ce_loss'))
            suffix_loss = loss.new_zeros(())
            kd_indices = [i for i,(record,side,mode) in enumerate(batch)
                          if mode and (kd_directions is None or (record['task_id'],side) in kd_directions)]
            gold_indices = [i for i,(_,_,mode) in enumerate(batch) if not mode]
            traces = [prepared_traces[(record['task_id'],side,mode)] for record,side,mode in batch]
            student_suffixes = [part.suffix_ids for part in targets]
        else:
            with torch.no_grad():
                if frozen_training_states is not None:
                    pasts,cached_suffixes,traces,states,teacher = static_adapter_states.load_batch(
                        frozen_training_states,batch,base.input_device(model))
                    teacher_pasts = teacher_suffixes = suffix_teacher_pasts = suffix_teacher_seeds = None
                elif paired_context is not None:
                    traces = ([prepared_traces[(record['task_id'],side,mode)] for record,side,mode in batch]
                              if args.paired_role_forward else
                              mixed_traces_batch(model,tokenizer,batch,None,None,args.teacher_tokens,
                                                 trace_cache=trace_cache))
                    pasts,states,teacher = paired_context.get_batch(
                        targets,sources,traces,[mode for _,_,mode in batch],
                        tensorized=args.paired_role_forward)
                    teacher_pasts = teacher_suffixes = suffix_teacher_pasts = suffix_teacher_seeds = None
                elif prefix_cache is not None:
                    prefix_layers, states = prefix_cache.get_batch([part.prefix_ids for part in targets])
                    pasts = reuse_pasts_batch(model,targets,sources,prefix_layers=prefix_layers,shared_rope=True)
                else:
                    pasts=reuse_pasts_batch(model,targets,sources)
                if frozen_training_states is not None or paired_context is not None:
                    pass
                elif args.single_full_teacher_forward:
                    teacher_pasts = teacher_suffixes = suffix_teacher_pasts = suffix_teacher_seeds = None
                elif args.mixed_gold_full:
                    teacher_pasts = teacher_suffixes = suffix_teacher_pasts = suffix_teacher_seeds = None
                    traces, teacher = mixed_supervision(model, tokenizer, batch, targets, args.teacher_tokens)
                elif args.teacher_mode == "full":
                    teacher_pasts, teacher_suffixes, suffix_teacher_pasts, suffix_teacher_seeds = (
                        full_teacher_contexts_batch(model, targets, bool(args.suffix_kl_weight))
                    )
                else:
                    teacher_pasts, teacher_suffixes = pasts, [part.suffix_ids for part in repeated]
                    suffix_teacher_pasts = suffix_teacher_seeds = None
                if frozen_training_states is not None or paired_context is not None:
                    pass
                elif args.mixed_gold_full:
                    pass
                elif args.teacher_trace == "mixed":
                    traces = mixed_traces_batch(
                        model, tokenizer, batch, teacher_pasts, teacher_suffixes, args.teacher_tokens,
                        trace_cache=trace_cache,
                    )
                elif args.teacher_trace == "gold":
                    traces = gold_traces_batch(tokenizer, batch, base.input_device(model))
                else:
                    traces=generate_teachers_batch(
                        model, tokenizer, teacher_pasts, teacher_suffixes, args.teacher_tokens,
                        stop_at_closing_brace=[
                            not reasoning and record.get("output_protocol") != "json_answer"
                            for record, _, reasoning in batch
                        ],
                        no_reasoning_tokens=args.no_reasoning_teacher_tokens,
                    )
                if frozen_training_states is not None or paired_context is not None:
                    pass
                elif args.single_full_teacher_forward:
                    reasoning_indices = [i for i,(_,_,reasoning) in enumerate(batch) if reasoning]
                    reference = full_teacher_logits_batch(model,[targets[i] for i in reasoning_indices],
                                                     [traces[i] for i in reasoning_indices])
                    teacher = [None]*len(batch)
                    for i,prediction in zip(reasoning_indices,reference):
                        teacher[i] = prediction
                elif not args.mixed_gold_full:
                    teacher=continuation_logits_batch(model,list(zip(teacher_pasts,teacher_suffixes,traces)),None)
                if args.suffix_kl_weight:
                    assert suffix_teacher_pasts is not None and suffix_teacher_seeds is not None
                    suffix_teacher = continuation_logits_batch(
                        model,
                        list(zip(suffix_teacher_pasts, suffix_teacher_seeds, [part.suffix_ids for part in targets])),
                        None,
                    )
                if frozen_training_states is None and prefix_cache is None and paired_context is None:
                    states=target_prefix_states_batch(
                        model, [part.prefix_ids for part in targets], pooling=args.prefix_pooling,
                    )
            latents=adapter(states)
            student_suffixes = cached_suffixes if frozen_training_states is not None else [part.suffix_ids for part in targets]
            if isinstance(pasts,BatchedPast):
                student=continuation_logits_batch(
                    model,list(zip([None]*len(batch),student_suffixes,traces)),latents,
                    batched_past=pasts)
            else:
                student=continuation_logits_batch(model,list(zip(pasts,student_suffixes,traces)),latents)
            kd_indices = [
                index for index, (record, side, reasoning) in enumerate(batch)
                if not args.mixed_gold_full and (kd_directions is None or (record["task_id"], side) in kd_directions)
                and (args.teacher_trace != "mixed" or reasoning)
            ]
            answer_loss = masked_answer_kd(teacher, student, kd_indices, args.temperature)
            gold_indices = [
                index for index, (_, _, reasoning) in enumerate(batch)
                if args.gold_first_token_ce_weight and (args.teacher_trace != "mixed" or not reasoning)
            ]
            if args.gold_first_token_ce_weight:
                if args.gold_first_token_ce_vocab == "full":
                    gold_ce_loss = masked_gold_first_token_ce(student, traces, gold_indices)
                else:
                    option_token_ids = []
                    option_targets = []
                    for record, side, _ in batch:
                        option_count = len(record["metadata"]["option_values"])
                        letters = [chr(ord("A") + index) for index in range(option_count)]
                        token_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
                        if any(len(ids) != 1 for ids in token_ids):
                            raise ValueError("option-restricted CE requires single-token option letters")
                        option_token_ids.append(torch.tensor(
                            [ids[0] for ids in token_ids], dtype=torch.long,
                            device=base.input_device(model),
                        ))
                        option_targets.append(letters.index(record[f"gold_{side}"]))
                    gold_ce_loss=torch.stack([
                        torch.nn.functional.cross_entropy(
                            prediction[:1, ids].float(),
                            torch.tensor([target], dtype=torch.long, device=prediction.device),
                        )
                        for prediction, ids, target in zip(student, option_token_ids, option_targets)
                    ]).mean()
            else:
                gold_ce_loss=answer_loss.new_zeros(())
            if args.suffix_kl_weight:
                suffix_student = suffix_logits_batch(
                    model, list(zip(pasts, [part.suffix_ids for part in targets])), latents,
                )
                suffix_loss = torch.stack([
                    kl(reference, prediction, args.temperature)
                    for reference, prediction in zip(suffix_teacher, suffix_student)
                ]).mean()
            else:
                suffix_loss = answer_loss.new_zeros(())
            loss = args.kd_weight * answer_loss + args.gold_first_token_ce_weight * gold_ce_loss + args.suffix_kl_weight * suffix_loss
            if args.mixed_gold_full:
                loss, gold_ce_loss, answer_loss = mixed_supervision_loss(
                    student, teacher, traces, [reasoning for _, _, reasoning in batch], args.temperature)
                kd_indices = [i for i, (_, _, reasoning) in enumerate(batch) if reasoning]
            loss=loss+1e-4*(latents.square().mean()+adapter.base.square().mean()); loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0)); optimizer.step()
        if args.audit_task_loss:
            for record,side,mode,value in audit_rows:
                if not math.isfinite(value):raise ValueError('nonfinite per-task loss')
                key=(record['dataset'],record.get('metadata',{}).get('scene',record['dataset']),mode)
                stat=task_loss_totals.setdefault(key,{'sum':0.,'count':0,'pairs':set()})
                stat['sum']+=value;stat['count']+=1;stat['pairs'].add((record['task_id'],side))
        row={"step":step,"loss":float(loss.detach()),"answer_loss":float(answer_loss.detach()),"gold_ce_loss":float(gold_ce_loss.detach()),"kd_batch_directions":len(kd_indices),"suffix_loss":float(suffix_loss.detach()),"grad_norm":norm,"task_family":family,"batch_size":len(batch),"datasets":[record["dataset"] for record,_,_ in batch],"task_ids":[record["task_id"] for record,_,_ in batch],"targets":[side for _,side,_ in batch],"reasoning":[reasoning for _,_,reasoning in batch],"teacher_tokens":[int(trace.numel()) for trace in traces],"suffix_tokens":[int(suffix.numel()) for suffix in student_suffixes],"elapsed_seconds":time.perf_counter()-started}
        row["cuda_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
        if args.paired_role_forward and args.gold_first_token_ce_vocab == 'full':
            row['microbatch_size'] = actual_microbatch
            row['oom_retries'] = oom_retries
        if paired_context is not None:
            row['paired_role_context'] = {'dense_forward_calls':paired_context.forward_calls,
                                         'retained_dataset_cache_bytes':paired_context.bytes}
        if prefix_cache is not None:
            row['prefix_cache'] = {'hits':prefix_cache.hits,'misses':prefix_cache.misses,
                                   'gib':prefix_cache.bytes/2**30}
        if args.mixed_gold_full:
            row["supervision"] = ["full_kl" if reasoning else "gold_sequence_ce" for _, _, reasoning in batch]
            row["gold_ce_batch_directions"] = sum(not reasoning for _, _, reasoning in batch)
        if args.teacher_trace == "mixed":
            row["supervision"] = ["full_generated_kd" if reasoning else "gold_first_token_ce"
                                  for _, _, reasoning in batch]
            row["gold_ce_batch_directions"] = len(gold_indices)
            row["supervision_counts"] = {"gold_first_token_ce": len(gold_indices),
                                         "full_generated_kd": len(kd_indices)}
        row["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        base.append_jsonl(history,row); progress.set_postfix(loss=f"{row['loss']:.4f}",batch=len(batch),tokens=max(row["teacher_tokens"]))
        completed_steps = step
        epoch_losses.append(row["loss"])
        if args.latest_save_interval and step % args.latest_save_interval == 0:
            torch.save({"adapter":adapter.state_dict(),"config":config,"step":step}, args.output / "latest_prefix_adapter.pt")
        if step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            if args.audit_task_loss:
                metrics=[{'dataset':key[0],'scene':key[1],'mode':key[2],
                          'mean_loss':stat['sum']/stat['count'],'directions':stat['count'],
                          'unique_directions':len(stat['pairs'])} for key,stat in sorted(task_loss_totals.items())]
                for metric in metrics:
                    expected=2*sum(r['dataset']==metric['dataset'] and
                        r.get('metadata',{}).get('scene',r['dataset'])==metric['scene'] for r in train)
                    if metric['directions']!=expected or metric['unique_directions']!=expected:
                        raise ValueError('task loss audit missing/duplicate directions')
                base.write_json_atomic(args.output/f'epoch_{epoch:03d}_task_loss.json',{'epoch':epoch,'tasks':metrics})
                task_loss_totals={}
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            if args.save_epoch_checkpoints:
                torch.save({"adapter":adapter.state_dict(),"config":config,"step":step,"epoch":epoch,
                            "epoch_mean_loss":epoch_loss}, args.output / f"epoch_{epoch:03d}_prefix_adapter.pt")
            improved = epoch_loss < best_epoch_loss - args.early_stopping_min_delta
            if improved:
                best_epoch_loss = epoch_loss
                epochs_without_improvement = 0
                torch.save({"adapter":adapter.state_dict(),"config":config,"step":step,"epoch":epoch,"epoch_mean_loss":epoch_loss}, args.output / "best_prefix_adapter.pt")
            else:
                epochs_without_improvement += 1
            base.append_jsonl(history,{"event":"epoch_end","epoch":epoch,"step":step,"mean_loss":epoch_loss,"best_mean_loss":best_epoch_loss,"epochs_without_improvement":epochs_without_improvement,"improved":improved,"elapsed_seconds":time.perf_counter()-started})
            print(f"[epoch {epoch}] mean_loss={epoch_loss:.6f} best={best_epoch_loss:.6f} stagnant={epochs_without_improvement}", flush=True)
            epoch_losses = []
            if args.early_stopping_patience_epochs and epochs_without_improvement >= args.early_stopping_patience_epochs:
                print(f"[stop] no epoch-loss improvement for {epochs_without_improvement} epochs", flush=True)
                break
        if args.paired_role_forward and args.gold_first_token_ce_vocab == 'full':
            del values, traces, loss
        else:
            del pasts,teacher_pasts,traces,teacher,states,latents,student,loss
    checkpoint=args.output / "latest_prefix_adapter.pt"; torch.save({"adapter":adapter.state_dict(),"config":config,"step":completed_steps},checkpoint)
    base.write_json_atomic(args.output / "summary.json",{"complete":completed_steps == args.steps,"completed_steps":completed_steps,"completed_epochs":completed_steps // steps_per_epoch,"best_epoch_mean_loss":best_epoch_loss if best_epoch_loss < float("inf") else None,"config":config,"checkpoint":str(checkpoint),"best_checkpoint":str(args.output / "best_prefix_adapter.pt"),"wall_seconds":time.perf_counter()-started})
    return 0

if __name__ == "__main__": raise SystemExit(main())
