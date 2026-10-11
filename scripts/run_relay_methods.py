#!/usr/bin/env python3
"""Run RelayCaching, CacheBlend, or EPIC on the project's A/B reuse tasks.

This is a *static-block adapter* for RelayCaching.  The upstream ``agent``
message is the already-written ``shared_block``: it is prefetched under the
source side's prefix, sliced from that real source-context cache, and then
provided to RelayCaching as ``{agent_<source>_current}`` in the target prompt.
Thus no text is regenerated merely to manufacture an upstream message.

Only the requested method is executed.  Existing full-prefill and direct
cross-reuse results remain the references; this runner deliberately does not
rerun them.

Run with the dedicated environment, for example:

    PYTHONPATH=third_party/RelayCaching .venv-relay/bin/python \
    scripts/run_relay_methods.py --method cacheblend --model 1.7b --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import os
import statistics
import sys
import time
import traceback
import uuid
from types import MethodType
from pathlib import Path
from typing import Any

print("[startup] importing torch", flush=True)
import torch
print(f"[startup] torch imported: {torch.__version__}", flush=True)

ROOT = Path(__file__).resolve().parents[1]
RELAY_ROOT = ROOT / "third_party" / "RelayCaching"
if str(RELAY_ROOT) not in sys.path:
    sys.path.insert(0, str(RELAY_ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

print("[startup] importing RelayCaching adapters", flush=True)
from relay_deal_adapter import BOXED_PREFILL, SYSTEM_MESSAGE, iter_examples, load_jsonl, parse_prediction, write_metadata
from RelayCaching.llm.config import RelayCachingConfig
from RelayCaching.llm.gpt_chat import LLMChat
from RelayCaching.utils.relay_utils import BatchedRelayCache
from transformers import DynamicCache
print("[startup] RelayCaching adapters imported", flush=True)


MODELSCOPE_CACHE = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
MODEL_PATHS = {
    "0.6b": str(MODELSCOPE_CACHE / "models/Qwen--Qwen3-0.6B/snapshots/master"),
    "1.7b": str(MODELSCOPE_CACHE / "models/Qwen--Qwen3-1.7B/snapshots/master"),
    "4b": str(MODELSCOPE_CACHE / "models/Qwen--Qwen3-4B/snapshots/master"),
    "8b": str(MODELSCOPE_CACHE / "models/Qwen--Qwen3-8B/snapshots/master"),
}
METHODS = ("relaycaching", "cacheblend", "epic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--model", default="1.7b", choices=sorted(MODEL_PATHS))
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/benchmark/benchmark_final240_real_ood_330_20261006.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/relay_methods")
    parser.add_argument("--limit", type=int, default=None, help="Limit original records, before two directions.")
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of input records per requested batch. All selected directions are included; CacheBlend and static RelayCaching use a two-direction stability cap.",
    )
    parser.add_argument("--warmup-batches", type=int, default=0,
                        help="Run untimed static batches before timing inference.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--explicit-reasoning", action="store_true",
        help="Ask for visible step-by-step reasoning followed by exactly one boxed answer."
    )
    parser.add_argument("--directions", nargs="+", choices=("a_to_b", "b_to_a"), default=("a_to_b", "b_to_a"))
    parser.add_argument(
        "--use-debug-selection-direction", action="store_true",
        help="For selector-produced debug JSONL, evaluate only each row's debug_selection.condition."
    )
    parser.add_argument("--cacheblend-diff-por", type=float, default=0.2)
    parser.add_argument(
        "--execution-path", choices=("official-serial", "static-batched"), default="official-serial",
        help=(
            "official-serial uses the upstream RelayCaching generation path and its selective token logic; "
            "static-batched is a throughput-oriented diagnostic adapter and is not valid for paper TTFT claims."
        ),
    )
    parser.add_argument(
        "--cacheblend-bucket-by-length", action="store_true",
        help="Group CacheBlend batch directions by target prompt length before microbatching.",
    )
    parser.add_argument("--epic-prefix-tokens", type=int, default=16)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--det-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=-1)
    parser.add_argument("--tau-dev", type=float, default=0.0)
    parser.add_argument("--tau-inf", type=float, default=0.0)
    parser.add_argument("--suffix-tokens", type=int, default=0,
                        help="Number of trailing relay tokens always selected for correction.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Discard existing samples and rerun every direction instead of resuming.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be non-negative")
    if args.epic_prefix_tokens < 0:
        parser.error("--epic-prefix-tokens must be non-negative")
    if args.suffix_tokens < 0:
        parser.error("--suffix-tokens must be non-negative")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.execution_path == "official-serial" and (args.batch_size != 1 or args.warmup_batches != 0):
        parser.error("official-serial requires --batch-size 1 and --warmup-batches 0; "
                     "use --execution-path static-batched for batching")
    if args.method != "epic" and args.epic_prefix_tokens != 16:
        parser.error("--epic-prefix-tokens is used only by --method epic")
    if args.method not in {"epic", "cacheblend", "relaycaching"} and args.batch_size != 1:
        parser.error("true batching is currently implemented only for RelayCaching repository methods")
    return args


def _install_project_qwen_template(chat: LLMChat, *, assistant_prefill: str | None) -> None:
    """Use the *same* Qwen3 rendering contract as ``run_direct_reuse.py``.

    RelayCaching normally appends a string after an assistant generation header.
    Our no-reasoning experiment instead has an unfinished assistant message
    (``The answer is: \\boxed{``), so that the first generated token attends to
    both the reused block and this suffix.  ``continue_final_message`` is the
    tokenizer-supported way to express that contract.
    """
    def build(self: LLMChat, messages: Any, assistant_prompt: str | None = None,
              add_generation_prompt: bool = True):
        normalised = self._normalise_messages(messages)
        if assistant_prefill is None:
            rendered = self.tokenizer.apply_chat_template(
                normalised, tokenize=False, add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        else:
            rendered = self.tokenizer.apply_chat_template(
                normalised + [{"role": "assistant", "content": assistant_prefill}],
                tokenize=False, add_generation_prompt=False,
                continue_final_message=True, enable_thinking=False,
            )
        ids = self.tokenizer.encode(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        return inputs, rendered, int(ids.shape[-1])
    chat._build_chat_inputs = MethodType(build, chat)


def _slice_shared_block_cache(
    source: LLMChat,
    *,
    source_prefix: str,
    shared_block: str,
) -> tuple[Any, dict[str, torch.Tensor], dict[str, int], torch.Tensor | None]:
    """Materialise block KV under the source prefix and convert it to relative RoPE.

    The block boundaries are computed from the model's own chat-template token
    streams, rather than by separately tokenizing text fragments.  This avoids
    silently changing Qwen's whitespace/BPE boundary tokens.
    """
    full_messages = source._render_base_messages(
        SYSTEM_MESSAGE, f"{source_prefix}\n\n{shared_block}"
    )
    full_inputs, rendered, _ = source._build_chat_inputs(full_messages, add_generation_prompt=True)
    user_text = f"{source_prefix}\n\n{shared_block}"
    user_start = rendered.find(user_text)
    if user_start < 0:
        raise RuntimeError("Qwen chat template did not preserve source user text verbatim")
    char_start = user_start + len(source_prefix) + 2
    char_end = char_start + len(shared_block)
    encoded = source.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    token_ids = encoded["input_ids"]
    if token_ids != full_inputs["input_ids"][0].tolist():
        raise RuntimeError("offset tokenizer and source prefill tokenizer disagree")
    # Same token-boundary rule as run_direct_reuse.py: retain an end-crossing
    # token with the reusable block, but keep a start-crossing token in prefix.
    nonempty = [(i, start, end) for i, (start, end) in enumerate(offsets) if end > start]
    starts = [i for i, start, _ in nonempty if start >= char_start]
    ends = [i for i, start, _ in nonempty if start >= char_end]
    if not starts:
        raise RuntimeError("shared block has no token-aligned start")
    block_start = starts[0]
    block_end = ends[0] if ends else len(token_ids)
    if not (0 <= block_start < block_end <= full_inputs["input_ids"].shape[-1]):
        raise RuntimeError(
            "Could not isolate shared_block in the Qwen chat template: "
            f"start={block_start}, end={block_end}, full={full_inputs['input_ids'].shape[-1]}"
        )
    captured_hidden: list[torch.Tensor] = []

    def capture_layer_input(_module: Any, inputs: tuple[Any, ...]) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            captured_hidden.append(inputs[0].detach().clone())

    layer_index = int(source.config.start_layer)
    if not 0 <= layer_index < len(source.model.model.layers):
        raise ValueError(f"invalid RelayCaching start layer: {layer_index}")
    hook = source.model.model.layers[layer_index].register_forward_pre_hook(capture_layer_input)
    try:
        with torch.no_grad():
            out = source.model(**full_inputs, use_cache=True, return_dict=True)
    finally:
        hook.remove()
    if len(captured_hidden) != 1:
        raise RuntimeError(f"expected one hidden-state capture at layer {layer_index}, got {len(captured_hidden)}")
    cache = out.past_key_values.copy().slice_(start=block_start, end=block_end)
    # RelayCaching stores upstream response KV at relative positions, then its
    # official path rotates it to the target placeholder position.
    cache = source.anchor_engine.apply_rotary_pos_emb(cache, offset=-block_start)
    ids = {
        "input_ids": full_inputs["input_ids"][:, block_start:block_end],
        "attention_mask": torch.ones_like(full_inputs["input_ids"][:, block_start:block_end]),
    }
    return cache, ids, {
        "source_block_start": block_start, "source_block_end": block_end,
        "source_block_char_start": char_start, "source_block_char_end": char_end,
    }, captured_hidden[0][:, block_start:block_end, :].contiguous()


def _store_static_agent_block(
    source: LLMChat,
    *,
    message: str,
    cache: Any,
    ids: dict[str, torch.Tensor],
    hidden_states: torch.Tensor | None,
    hidden_layer: int,
) -> None:
    """Populate the exact shared-memory slots used by agent placeholders."""
    memory = source._shared_kv_cache_memory[source.node_id]
    for key, value in (
        ("response", cache),
        ("response_ids", ids),
        ("response_drop_num", 0),
        ("hidden_states", hidden_states if hidden_states is not None else []),
        ("hidden_states_layer_idx", hidden_layer if hidden_states is not None else 0),
        ("additional_recompute_indices", []),
        ("additional_recompute_scores", []),
    ):
        memory.setdefault(key, {}).setdefault(message, []).append(value)


def _make_config(args: argparse.Namespace) -> RelayCachingConfig:
    return RelayCachingConfig(
        execution_mode=args.method,
        max_tokens=args.max_new_tokens,
        do_sample=False,
        start_layer=args.start_layer,
        det_layer=args.det_layer,
        end_layer=args.end_layer,
        tau_dev=args.tau_dev,
        tau_inf=args.tau_inf,
        suffix_tokens=args.suffix_tokens,
        cacheblend_diff_por=args.cacheblend_diff_por,
        epic_prefix_token_num=args.epic_prefix_tokens,
    ).validate()


def _token_span(tokenizer: Any, rendered: str, *, char_start: int, char_end: int) -> tuple[int, int]:
    """Return the same token-aligned block span used by the serial adapter."""
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    nonempty = [(index, start, end) for index, (start, end) in enumerate(encoded["offset_mapping"])
                if end > start]
    starts = [index for index, start, _ in nonempty if start >= char_start]
    ends = [index for index, start, _ in nonempty if start >= char_end]
    if not starts:
        raise RuntimeError("shared block has no token-aligned start")
    start = starts[0]
    end = ends[0] if ends else len(encoded["input_ids"])
    if end <= start:
        raise RuntimeError("shared block has no token-aligned cache span")
    return start, end


def _static_batch_parts(source: LLMChat, target: LLMChat, example: Any, *, explicit_reasoning: bool) -> dict[str, Any]:
    """Tokenize source and dense target prompts into prefix/block/suffix IDs.

    EPIC recomputes its leading block tokens under the target prefix.  Keeping
    the three token ranges separately lets that correction, the donor tail,
    and target suffix all be evaluated in batched forwards.
    """
    # Keep source extraction identical to the established serial adapter.
    # Its block end may include a boundary-crossing delimiter token.
    source_user = example.source_user_content()
    source_messages = source._render_base_messages(SYSTEM_MESSAGE, source_user)
    _, source_rendered, _ = source._build_chat_inputs(source_messages, add_generation_prompt=True)
    source_user_start = source_rendered.find(source_user)
    if source_user_start < 0:
        raise RuntimeError("Qwen chat template did not preserve source user text verbatim")
    source_char_start = source_user_start + len(example.source_prefix) + 2
    source_char_end = source_char_start + len(example.shared_block)
    source_start, source_end = _token_span(
        source.tokenizer, source_rendered, char_start=source_char_start, char_end=source_char_end
    )
    source_ids = torch.tensor(
        source.tokenizer(source_rendered, add_special_tokens=False)["input_ids"], dtype=torch.long
    )
    # ``prepare_prefix_kv_segments`` tokenizes the template text on each side of
    # the placeholder independently.  A dense prompt tokenization can merge
    # tokens across the same boundaries and is therefore not equivalent.
    target_template = example.target_user_template(explicit_reasoning=explicit_reasoning)
    target_messages = target._render_base_messages(SYSTEM_MESSAGE, target_template)
    _, target_rendered, _ = target._build_chat_inputs(target_messages, add_generation_prompt=True)
    _, _, segments = target.locate_placeholder(target_rendered, return_segments=True)
    placeholder_indices = [index for index, segment in enumerate(segments) if segment[0] == "placeholder"]
    if len(placeholder_indices) != 1:
        raise RuntimeError(f"{example.task_id}: expected exactly one RelayCaching placeholder")
    placeholder_index = placeholder_indices[0]
    if placeholder_index == 0 or placeholder_index == len(segments) - 1:
        raise RuntimeError(f"{example.task_id}: placeholder must have both prefix and suffix text")
    target_prefix_ids = torch.cat(
        [segment[2]["input_ids"].detach().cpu()[0] for segment in segments[:placeholder_index]]
    )
    target_suffix_ids = torch.cat(
        [segment[2]["input_ids"].detach().cpu()[0] for segment in segments[placeholder_index + 1:]]
    )
    return {
        "source_ids": source_ids,
        "source_start": source_start,
        "source_end": source_end,
        "target_prefix_ids": target_prefix_ids,
        "block_ids": source_ids[source_start:source_end].clone(),
        "suffix_ids": target_suffix_ids,
        "boundaries": {
            "source_block_start": source_start, "source_block_end": source_end,
            "source_block_char_start": source_char_start, "source_block_char_end": source_char_end,
        },
    }


@torch.inference_mode()
def _prefill_batch(model: Any, ids_list: list[torch.Tensor]) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, int]:
    """Run variable-length prompts through the decoder backbone once."""
    device = model.device
    lengths = torch.tensor([int(ids.numel()) for ids in ids_list], device=device)
    width = int(lengths.max().item())
    input_ids = torch.zeros((len(ids_list), width), dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    positions = torch.zeros_like(input_ids)
    for row, ids in enumerate(ids_list):
        length = int(ids.numel())
        input_ids[row, width - length:] = ids.to(device)
        attention[row, width - length:] = 1
        positions[row, width - length:] = torch.arange(length, device=device)
    output = model.model(
        input_ids=input_ids, attention_mask=attention, position_ids=positions,
        cache_position=torch.arange(width, device=device), use_cache=True, return_dict=True,
    )
    layers = [(key.detach(), value.detach()) for key, value in zip(
        output.past_key_values.key_cache, output.past_key_values.value_cache
    )]
    return layers, lengths, width


def _gather_span(tensor: torch.Tensor, sequence_lengths: torch.Tensor, starts: torch.Tensor,
                 lengths: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = int(lengths.max().item())
    relative = torch.arange(maximum, device=tensor.device)
    physical = width - sequence_lengths[:, None] + starts[:, None] + relative
    valid = relative[None, :] < lengths[:, None]
    index = physical.clamp(0, tensor.shape[-2] - 1)[:, None, :, None].expand(
        -1, tensor.shape[1], -1, tensor.shape[-1]
    )
    return torch.gather(tensor, -2, index).masked_fill(~valid[:, None, :, None], 0), valid


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values[..., :values.shape[-1] // 2], values[..., values.shape[-1] // 2:]
    return torch.cat((-second, first), dim=-1)


def _relocate_keys(model: Any, keys: torch.Tensor, valid: torch.Tensor,
                   source_starts: torch.Tensor, target_starts: torch.Tensor) -> torch.Tensor:
    """Move post-RoPE keys between row-specific absolute prompt positions."""
    length = keys.shape[-2]
    relative = torch.arange(length, device=keys.device)
    source_positions = source_starts[:, None] + relative
    target_positions = target_starts[:, None] + relative
    dummy = torch.zeros((keys.shape[0], 2 * length, keys.shape[-1]), dtype=keys.dtype, device=keys.device)
    cos, sin = model.model.rotary_emb(dummy, torch.cat((source_positions, target_positions), dim=1))
    cos, sin = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
    source_cos, target_cos = cos[..., :length, :], cos[..., length:, :]
    source_sin, target_sin = sin[..., :length, :], sin[..., length:, :]
    unrotated = keys.float() * source_cos - _rotate_half(keys.float()) * source_sin
    relocated = unrotated * target_cos + _rotate_half(unrotated) * target_sin
    return relocated.to(dtype=keys.dtype).masked_fill(~valid[:, None, :, None], 0)


def _make_batch_cache(model: Any, layers: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    try:
        cache = DynamicCache(config=model.config)
        for cache_layer, (key, value) in zip(cache.layers, layers):
            cache_layer.keys, cache_layer.values = key, value
            cache_layer.dtype, cache_layer.device, cache_layer.is_initialized = key.dtype, key.device, True
    except TypeError:
        # RelayCaching currently pins a Transformers release whose DynamicCache
        # has no ``config`` constructor parameter.
        cache = DynamicCache()
        for layer_index, (key, value) in enumerate(layers):
            cache.update(key, value, layer_index)
    return cache


def _append_donor_tail(model: Any, prefix_layers: list[tuple[torch.Tensor, torch.Tensor]],
                       prefix_lengths: torch.Tensor, source_layers: list[tuple[torch.Tensor, torch.Tensor]],
                       source_lengths: torch.Tensor, source_width: int, source_starts: torch.Tensor,
                       block_lengths: torch.Tensor, recomputed_lengths: torch.Tensor) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Append relocated unrecomputed donor tokens with two batch-wide scatters per layer."""
    tail_lengths = block_lengths - recomputed_lengths
    final_lengths = prefix_lengths + tail_lengths
    final_width = int(final_lengths.max().item())
    prefix_width = prefix_layers[0][0].shape[-2]
    max_tail = int(tail_lengths.max().item())
    prefix_rel = torch.arange(prefix_width, device=prefix_lengths.device)
    tail_rel = torch.arange(max_tail, device=prefix_lengths.device)
    prefix_valid = prefix_rel[None, :] >= (prefix_width - prefix_lengths[:, None])
    prefix_dest = torch.where(
        prefix_valid, prefix_rel[None, :] - (prefix_width - final_width), torch.full_like(prefix_rel[None, :], final_width)
    )
    tail_valid = tail_rel[None, :] < tail_lengths[:, None]
    tail_dest = torch.where(
        tail_valid, final_width - final_lengths[:, None] + prefix_lengths[:, None] + tail_rel[None, :],
        torch.full_like(tail_rel[None, :], final_width),
    )
    output = []
    for (prefix_key, prefix_value), (source_key, source_value) in zip(prefix_layers, source_layers):
        tail_key, valid = _gather_span(
            source_key, source_lengths, source_starts + recomputed_lengths, tail_lengths, source_width
        )
        tail_value, _ = _gather_span(
            source_value, source_lengths, source_starts + recomputed_lengths, tail_lengths, source_width
        )
        tail_key = _relocate_keys(
            model, tail_key, valid, source_starts + recomputed_lengths, prefix_lengths
        )
        shape = (prefix_key.shape[0], prefix_key.shape[1], final_width + 1, prefix_key.shape[-1])
        key, value = torch.zeros(shape, dtype=prefix_key.dtype, device=prefix_key.device), torch.zeros(shape, dtype=prefix_value.dtype, device=prefix_value.device)
        prefix_index = prefix_dest[:, None, :, None].expand(-1, key.shape[1], -1, key.shape[-1])
        tail_index = tail_dest[:, None, :, None].expand(-1, key.shape[1], -1, key.shape[-1])
        key.scatter_(2, prefix_index, prefix_key).scatter_(2, tail_index, tail_key)
        value.scatter_(2, prefix_index, prefix_value).scatter_(2, tail_index, tail_value)
        output.append((key[..., :final_width, :], value[..., :final_width, :]))
    return output, final_lengths


@torch.inference_mode()
def _forward_suffix_batch(model: Any, layers: list[tuple[torch.Tensor, torch.Tensor]],
                          prefix_lengths: torch.Tensor, suffix_ids: list[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    device = model.device
    suffix_lengths = torch.tensor([int(ids.numel()) for ids in suffix_ids], device=device)
    suffix_width, prefix_width = int(suffix_lengths.max().item()), layers[0][0].shape[-2]
    input_ids = torch.zeros((len(suffix_ids), suffix_width), dtype=torch.long, device=device)
    suffix_mask, positions = torch.zeros_like(input_ids), torch.zeros_like(input_ids)
    for row, ids in enumerate(suffix_ids):
        length = int(ids.numel())
        input_ids[row, :length] = ids.to(device)
        suffix_mask[row, :length] = 1
        positions[row, :length] = torch.arange(int(prefix_lengths[row]), int(prefix_lengths[row]) + length, device=device)
    attention = torch.zeros((len(suffix_ids), prefix_width + suffix_width), dtype=torch.long, device=device)
    for row, length in enumerate(prefix_lengths.tolist()):
        attention[row, prefix_width - length:prefix_width] = 1
        attention[row, prefix_width:] = suffix_mask[row]
    output = model.model(
        input_ids=input_ids, attention_mask=attention, position_ids=positions,
        cache_position=torch.arange(prefix_width, prefix_width + suffix_width, device=device),
        past_key_values=_make_batch_cache(model, layers), use_cache=True, return_dict=True,
    )
    logits = model.lm_head(output.last_hidden_state[
        torch.arange(len(suffix_ids), device=device), suffix_lengths - 1
    ]).float()
    totals = prefix_lengths + suffix_lengths
    compact_width = int(totals.max().item())
    relative = torch.arange(compact_width, device=device)[None, :] - (compact_width - totals[:, None])
    valid, physical = relative >= 0, prefix_width - prefix_lengths[:, None] + relative
    compact = []
    for key, value in zip(output.past_key_values.key_cache, output.past_key_values.value_cache):
        index = physical.clamp(0, key.shape[-2] - 1)[:, None, :, None].expand(-1, key.shape[1], -1, key.shape[-1])
        compact.append((torch.gather(key, -2, index).masked_fill(~valid[:, None, :, None], 0),
                        torch.gather(value, -2, index).masked_fill(~valid[:, None, :, None], 0)))
    return logits, compact, totals


@torch.inference_mode()
def _decode_batch(model: Any, tokenizer: Any, logits: torch.Tensor, layers: list[tuple[torch.Tensor, torch.Tensor]],
                  lengths: torch.Tensor, max_new_tokens: int, *, stop_after_boxed_answer: bool = False,
                  constrain_boxed_choice: bool = False,
                  allowed_choice_labels: list[set[str]] | None = None,
                  stop_after_boxed_choice: bool = False) -> list[list[int]]:
    device, batch, cache_width = model.device, logits.shape[0], layers[0][0].shape[-2]
    choice_token_ids: dict[str, int] = {}
    close_brace_token_id: int | None = None
    if constrain_boxed_choice:
        # The no-reasoning prompt is already inside ``\boxed{``. Constrain its
        # continuation to exactly one uppercase option label and the closing
        # brace. This is format enforcement, not an answer-selection heuristic.
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            encoded = tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) == 1:
                choice_token_ids[letter] = int(encoded[0])
        brace = tokenizer.encode("}", add_special_tokens=False)
        if not choice_token_ids or len(brace) != 1:
            raise RuntimeError("tokenizer cannot represent constrained boxed-answer tokens")
        close_brace_token_id = int(brace[0])
        mask = torch.full_like(logits, float("-inf"))
        labels_by_row = allowed_choice_labels or [set(choice_token_ids)] * batch
        if len(labels_by_row) != batch:
            raise ValueError("allowed_choice_labels must have one set per batch row")
        for row, labels in enumerate(labels_by_row):
            allowed_ids = [choice_token_ids[label] for label in labels if label in choice_token_ids]
            if not allowed_ids:
                raise RuntimeError(f"no tokenizable option labels for batch row {row}: {sorted(labels)}")
            mask[row, allowed_ids] = logits[row, allowed_ids]
        logits = mask
    cache, current = _make_batch_cache(model, layers), logits.argmax(dim=-1).view(batch, 1)
    tokens = [[int(current[row, 0])] for row in range(batch)]
    eos_ids = {tokenizer.eos_token_id} if isinstance(tokenizer.eos_token_id, int) else set(tokenizer.eos_token_id or [])
    pad_done_token = next(iter(eos_ids), 0)
    labels_by_row = allowed_choice_labels or [set(choice_token_ids)] * batch

    def completed(row: int) -> bool:
        text = tokenizer.decode(tokens[row], skip_special_tokens=True)
        if stop_after_boxed_answer and "}" in text:
            return True
        if stop_after_boxed_choice:
            # Reasoning is unconstrained, but its first complete boxed valid
            # option is the final answer. Stop before Qwen emits a synthetic
            # next ``Human:`` turn or repeats the chat template.
            match = re.search(r"\\boxed\{\s*([A-Z])\s*\}", text)
            return match is not None and match.group(1) in labels_by_row[row]
        return False

    done = torch.tensor([
        tokens[row][0] in eos_ids or completed(row)
        for row in range(batch)
    ], device=device)
    attention = torch.zeros((batch, cache_width), dtype=torch.long, device=device)
    for row, length in enumerate(lengths.tolist()):
        attention[row, cache_width - length:] = 1
    for step in range(1, max_new_tokens):
        if bool(done.all()):
            break
        was_done = done.clone()
        output = model(
            input_ids=current, attention_mask=torch.cat((attention, torch.ones((batch, step), dtype=torch.long, device=device)), dim=1),
            position_ids=torch.tensor([[int(lengths[row]) + step - 1] for row in range(batch)], device=device),
            cache_position=torch.tensor([cache_width + step - 1], device=device), past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        cache = output.past_key_values
        if constrain_boxed_choice and step == 1:
            current = torch.full((batch, 1), close_brace_token_id, dtype=torch.long, device=device)
        else:
            current = output.logits[:, -1].argmax(dim=-1).view(batch, 1)
        # Batched decoding continues until every row is complete. Feed EOS for
        # rows that were already complete, otherwise they keep sampling and can
        # leak into a synthetic ``Human:`` continuation while waiting for peers.
        current = torch.where(
            was_done[:, None],
            torch.full_like(current, pad_done_token),
            current,
        )
        for row, token in enumerate(current[:, 0].tolist()):
            tokens[row].append(token)
        done |= torch.tensor([
            token in eos_ids or completed(row)
            for row, token in enumerate(current[:, 0].tolist())
        ], device=device)
    return tokens


@torch.inference_mode()
def _correct_relay_batch(model: Any, layers: list[tuple[torch.Tensor, torch.Tensor]],
                         logical_lengths: torch.Tensor, prefix_lengths: torch.Tensor,
                         block_ids: list[torch.Tensor], recomputed: torch.Tensor,
                         args: argparse.Namespace) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Apply row-aware EPIC or CacheBlend correction to a static relay block."""
    device = model.device
    batch, cache_width = len(block_ids), layers[0][0].shape[-2]
    correction_width = int(recomputed.max().item())
    if correction_width == 0:
        return layers
    input_ids = torch.zeros((batch, correction_width), dtype=torch.long, device=device)
    query_valid = torch.zeros_like(input_ids, dtype=torch.bool)
    position_ids = torch.ones_like(input_ids)
    for row, block in enumerate(block_ids):
        length = int(recomputed[row])
        input_ids[row, :length] = block[:length].to(device)
        query_valid[row, :length] = True
        position_ids[row, :length] = torch.arange(
            int(prefix_lengths[row]), int(prefix_lengths[row]) + length, device=device
        )
    # The cache is physically left padded.  Index tensors therefore point at
    # physical slots, while position_ids above preserve logical RoPE positions.
    relative = torch.arange(correction_width, device=device)
    starts = cache_width - logical_lengths + prefix_lengths
    recompute_indices = starts[:, None] + relative[None, :]
    recompute_indices = recompute_indices.clamp(max=cache_width - 1)
    prefix_relative = torch.arange(cache_width, device=device)[None, :] - (cache_width - logical_lengths[:, None])
    blank_valid = (prefix_relative >= 0) & ~(
        (prefix_relative >= prefix_lengths[:, None])
        & (prefix_relative < prefix_lengths[:, None] + recomputed[:, None])
    )
    blank_indices = torch.arange(cache_width, device=device).expand(batch, -1)
    relay_indices = recompute_indices
    cache = _make_batch_cache(model, layers)
    relay_cache = BatchedRelayCache(
        cache, row_lengths=logical_lengths,
        recompute_indices=recompute_indices, recompute_valid=query_valid,
        relay_absolute_indices=relay_indices, relay_valid=query_valid,
        blank_absolute_indices=blank_indices, blank_valid=blank_valid,
        recompute_label=True, execution_mode=args.method,
        cacheblend_diff_por=args.cacheblend_diff_por,
    )
    attention = torch.zeros((batch, cache_width + correction_width), dtype=torch.long, device=device)
    for row, length in enumerate(logical_lengths.tolist()):
        attention[row, cache_width - length:cache_width] = 1
        attention[row, cache_width:cache_width + int(recomputed[row])] = 1
    # The row-aware cache ignores cache_position for correction writes; it is
    # supplied only for the Transformers model API. The hook builds its causal
    # rows from ``recompute_indices_by_row``.
    model.model(
        input_ids=input_ids, attention_mask=attention, position_ids=position_ids,
        cache_position=torch.arange(cache_width, cache_width + correction_width, device=device),
        past_key_values=relay_cache, use_cache=True, return_dict=True,
    )
    return [(key[..., :cache_width, :].contiguous(), value[..., :cache_width, :].contiguous())
            for key, value in zip(relay_cache.key_cache, relay_cache.value_cache)]


def _extend_cacheblend_layout(layers: list[tuple[torch.Tensor, torch.Tensor]], mixed_lengths: torch.Tensor,
                              target_prefix_lengths: torch.Tensor, block_lengths: torch.Tensor,
                              suffix_ids: list[torch.Tensor]) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand direct-reuse KV into CacheBlend's complete correction layout."""
    device = mixed_lengths.device
    suffix_lengths = torch.tensor([int(ids.numel()) for ids in suffix_ids], device=device)
    total_lengths = mixed_lengths + suffix_lengths
    total_width = int(total_lengths.max().item())
    old_width = layers[0][0].shape[-2]
    expanded = []
    for key, value in layers:
        out_key = torch.zeros((key.shape[0], key.shape[1], total_width, key.shape[-1]), dtype=key.dtype, device=device)
        out_value = torch.zeros_like(out_key)
        for row, length in enumerate(mixed_lengths.tolist()):
            old_start = old_width - length
            new_start = total_width - int(total_lengths[row])
            out_key[row, :, new_start:new_start + length] = key[row, :, old_start:old_start + length]
            out_value[row, :, new_start:new_start + length] = value[row, :, old_start:old_start + length]
        expanded.append((out_key, out_value))
    physical = total_width - total_lengths[:, None] + torch.arange(total_width, device=device)[None, :]
    valid = torch.arange(total_width, device=device)[None, :] < total_lengths[:, None]
    physical = torch.where(valid, physical, torch.zeros_like(physical))
    relay_start = total_width - total_lengths + target_prefix_lengths
    max_block = int(block_lengths.max().item())
    relay_indices = relay_start[:, None] + torch.arange(max_block, device=device)[None, :]
    relay_valid = torch.arange(max_block, device=device)[None, :] < block_lengths[:, None]
    relay_indices = torch.where(relay_valid, relay_indices, torch.zeros_like(relay_indices))
    return expanded, total_lengths, physical, relay_indices


@torch.inference_mode()
def _cacheblend_correct_batch(model: Any, layers: list[tuple[torch.Tensor, torch.Tensor]],
                              mixed_lengths: torch.Tensor, target_prefix_lengths: torch.Tensor,
                              block_ids: list[torch.Tensor], suffix_ids: list[torch.Tensor],
                              args: argparse.Namespace) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Run CacheBlend correction then append the final prompt token per row."""
    device = model.device
    if any(ids.numel() < 1 for ids in suffix_ids):
        raise ValueError("CacheBlend requires a final suffix token")
    block_lengths = torch.tensor([int(ids.numel()) for ids in block_ids], device=device)
    suffix_lengths = torch.tensor([int(ids.numel()) for ids in suffix_ids], device=device)
    total_lengths = mixed_lengths + suffix_lengths
    correction_lengths = total_lengths - 1
    width = int(correction_lengths.max().item())
    old_width = layers[0][0].shape[-2]
    full_layers = []
    for key, value in layers:
        full_key = torch.zeros((key.shape[0], key.shape[1], width, key.shape[-1]), dtype=key.dtype, device=device)
        full_value = torch.zeros_like(full_key)
        for row, source_length in enumerate(mixed_lengths.tolist()):
            source_start = old_width - source_length
            target_start = width - int(correction_lengths[row])
            full_key[row, :, target_start:target_start + source_length] = key[row, :, source_start:source_start + source_length]
            full_value[row, :, target_start:target_start + source_length] = value[row, :, source_start:source_start + source_length]
        full_layers.append((full_key, full_value))
    query_physical = width - correction_lengths[:, None] + torch.arange(width, device=device)[None, :]
    query_valid_full = torch.arange(width, device=device)[None, :] < correction_lengths[:, None]
    query_physical = torch.where(query_valid_full, query_physical, torch.zeros_like(query_physical))
    relay_start = width - correction_lengths + target_prefix_lengths
    relay_indices = relay_start[:, None] + torch.arange(int(block_lengths.max().item()), device=device)[None, :]
    batch = len(block_ids)
    correction_width = width
    input_ids = torch.zeros((batch, correction_width + 1), dtype=torch.long, device=device)
    query_valid = torch.zeros((batch, correction_width), dtype=torch.bool, device=device)
    position_ids = torch.ones_like(input_ids)
    for row, (block, suffix) in enumerate(zip(block_ids, suffix_ids)):
        prefix_len, block_len = int(target_prefix_lengths[row]), int(block_lengths[row])
        correction_len = int(correction_lengths[row])
        query_valid[row, :correction_len] = True
        position_ids[row, :correction_len] = torch.arange(correction_len, device=device)
        input_ids[row, prefix_len:prefix_len + block_len] = block.to(device)
        if suffix.numel() > 1:
            input_ids[row, prefix_len + block_len:correction_len] = suffix[:-1].to(device)
        input_ids[row, correction_width] = suffix[-1].to(device)
        position_ids[row, correction_width] = total_lengths[row] - 1
    # Caller attaches target-prefix ids; use an explicit attribute to avoid a
    # second mutable global data structure.
    prefix_ids = getattr(_cacheblend_correct_batch, "prefix_ids")
    for row, prefix in enumerate(prefix_ids):
        input_ids[row, :int(prefix.numel())] = prefix.to(device)
    blank_indices = torch.arange(width, device=device).expand(batch, -1)
    relay_valid = torch.arange(relay_indices.shape[1], device=device)[None, :] < block_lengths[:, None]
    cache = BatchedRelayCache(
        _make_batch_cache(model, full_layers), row_lengths=total_lengths,
        recompute_indices=query_physical, recompute_valid=query_valid,
        relay_absolute_indices=relay_indices, relay_valid=relay_valid,
        blank_absolute_indices=blank_indices, blank_valid=query_valid,
        query_physical_indices=query_physical, query_valid=query_valid,
        append_physical_indices=torch.full((batch, 1), width, dtype=torch.long, device=device),
        append_valid=torch.ones((batch, 1), dtype=torch.bool, device=device),
        recompute_label=True, execution_mode="cacheblend", cacheblend_diff_por=args.cacheblend_diff_por,
    )
    attention = torch.zeros((batch, width + 1), dtype=torch.long, device=device)
    for row, length in enumerate(correction_lengths.tolist()):
        attention[row, width - length:width] = 1
        attention[row, width] = 1
    output = model.model(
        input_ids=input_ids, attention_mask=attention, position_ids=position_ids,
        cache_position=torch.arange(correction_width, correction_width + 1, device=device), past_key_values=cache,
        use_cache=True, return_dict=True,
    )
    logits = model.lm_head(output.last_hidden_state[:, -1]).float()
    compact = [(key[..., :width + 1, :].contiguous(), value[..., :width + 1, :].contiguous())
               for key, value in zip(cache.key_cache, cache.value_cache)]
    return logits, compact, total_lengths


def _run_static_relay_batch(source: LLMChat, target: LLMChat, examples: list[Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """True batched EPIC/CacheBlend for static shared blocks."""
    phase_seconds: dict[str, float] = {}

    def timed(name: str, operation: Any) -> Any:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        began = time.perf_counter()
        value = operation()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        phase_seconds[name] = phase_seconds.get(name, 0.0) + time.perf_counter() - began
        return value

    began = time.perf_counter()
    parts = [_static_batch_parts(source, target, example, explicit_reasoning=args.explicit_reasoning) for example in examples]
    phase_seconds["prompt_build"] = time.perf_counter() - began
    model = target.model
    source_layers, source_lengths, source_width = timed("source_prefill", lambda: _prefill_batch(model, [part["source_ids"] for part in parts]))
    target_layers, target_lengths, _ = timed("target_prefix_prefill", lambda: _prefill_batch(model, [part["target_prefix_ids"] for part in parts]))
    device = model.device
    block_lengths = torch.tensor([int(part["block_ids"].numel()) for part in parts], device=device)
    recomputed = (
        torch.clamp(block_lengths, max=args.epic_prefix_tokens)
        if args.method == "epic" else block_lengths
    )
    source_starts = torch.tensor([part["source_start"] for part in parts], device=device)
    # First materialise the direct-reuse cache, then perform the method's
    # row-aware correction over the relay tokens.
    mixed_layers, mixed_lengths = timed("cache_assembly", lambda: _append_donor_tail(
        model, target_layers, target_lengths, source_layers, source_lengths, source_width,
        source_starts, block_lengths, torch.zeros_like(recomputed),
    ))
    if args.method in {"cacheblend", "relaycaching"}:
        _cacheblend_correct_batch.prefix_ids = [part["target_prefix_ids"] for part in parts]
        # This packed forward recomputes the relay region *and* consumes the
        # target suffix (the final suffix token is appended to produce its
        # first-token logits). It cannot be split into independent GPU
        # forwards without changing CacheBlend/RelayCaching semantics.
        logits, final_layers, final_lengths = timed("joint_repair_and_target_suffix", lambda: _cacheblend_correct_batch(
            model, mixed_layers, mixed_lengths, target_lengths,
            [part["block_ids"] for part in parts], [part["suffix_ids"] for part in parts], args,
        ))
    else:
        mixed_layers = timed("repair_recompute", lambda: _correct_relay_batch(
            model, mixed_layers, mixed_lengths, target_lengths,
            [part["block_ids"] for part in parts], recomputed, args,
        ))
        logits, final_layers, final_lengths = timed("target_suffix_prefill_and_first_token", lambda: _forward_suffix_batch(model, mixed_layers, mixed_lengths, [part["suffix_ids"] for part in parts]))
    allowed_choice_labels = [
        {
            label
            for dotted, bracketed in re.findall(
                r"(?m)^\s*(?:([A-Z])\.\s|\[([A-Z])\])",
                f"{example.shared_block}\n{example.question}",
            )
            for label in (dotted, bracketed)
            if label
        }
        for example in examples
    ]
    generated = timed("decode", lambda: _decode_batch(
        model, target.tokenizer, logits, final_layers, final_lengths, args.max_new_tokens,
        stop_after_boxed_answer=not args.explicit_reasoning,
        constrain_boxed_choice=not args.explicit_reasoning,
        allowed_choice_labels=allowed_choice_labels,
        stop_after_boxed_choice=args.explicit_reasoning,
    ))
    log_probs = torch.log_softmax(logits, dim=-1)
    top_log_probs, top_token_ids = torch.topk(log_probs, k=64, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    rows = []
    for row, (example, part, token_ids) in enumerate(zip(examples, parts, generated)):
        text = target.tokenizer.decode(token_ids, skip_special_tokens=True)
        if not args.explicit_reasoning and "}" in text:
            text = text[:text.index("}") + 1]
        completed_text = text if args.explicit_reasoning else BOXED_PREFILL + text
        prediction = parse_prediction(example.dataset, completed_text)
        rows.append({
            **example.as_metadata(explicit_reasoning=args.explicit_reasoning),
            "method": args.method, "prediction": prediction,
            "correct": prediction == str(example.gold).strip().upper(),
            "output": text, "output_with_prefill": completed_text,
            "elapsed_s": None, "relay_metadata": {
                "baseline_label": args.method, "algorithm_execution_mode": args.method,
                "placeholder_ids": [f"agent_{example.source}_current"],
                "ph_token_nums": int(block_lengths[row]), "recompute_num": int(recomputed[row]),
                "method_type": f"{args.method}_kv_reuse", "batched_static_adapter": True,
                "first_token_distribution": {
                    "entropy_nats": float(entropy[row]),
                    "top_token_ids": [int(value) for value in top_token_ids[row].tolist()],
                    "top_log_probs": [float(value) for value in top_log_probs[row].tolist()],
                },
            }, **part["boundaries"],
        })
    return rows, phase_seconds


async def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("RelayCaching evaluation requires a CUDA GPU; submit this command to a GPU node.")
    print(f"[setup] loading input: {args.input}", flush=True)
    records = load_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No records loaded from {args.input}")
    if args.use_debug_selection_direction:
        examples = []
        for record in records:
            direction = record.get("debug_selection", {}).get("condition")
            if direction not in {"a_to_b", "b_to_a"}:
                raise ValueError(
                    "--use-debug-selection-direction requires every record to contain "
                    "debug_selection.condition equal to a_to_b or b_to_a"
                )
            examples.extend(iter_examples([record], directions=(direction,)))
    else:
        examples = list(iter_examples(records, directions=args.directions))
    # Keep visible-reasoning outputs separate from the established no-reasoning
    # baseline folders, even when callers reuse the same output root.
    output_base = args.output_root / "reasoning" if args.explicit_reasoning else args.output_root
    output_dir = output_base / args.method / f"qwen3-{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "samples.jsonl"
    run_config_path = output_dir / "run_config.json"
    requested_direction_batch_size = args.batch_size * len(args.directions)
    run_config = {
        "method": args.method,
        "model": args.model,
        "model_path": MODEL_PATHS[args.model],
        "input": str(args.input),
        "records": len(records),
        "evaluated_directions": len(examples),
        "directions": list(args.directions),
        "batch_size": args.batch_size,
        "execution_path": args.execution_path,
        "warmup_batches": args.warmup_batches,
        "requested_direction_batch_size": requested_direction_batch_size,
        "effective_batch_size": (
            requested_direction_batch_size if args.execution_path == "static-batched" else 1
        ),
        "max_new_tokens": args.max_new_tokens,
        "explicit_reasoning": args.explicit_reasoning,
        "cacheblend_diff_por": args.cacheblend_diff_por,
        "cacheblend_bucket_by_length": args.cacheblend_bucket_by_length,
        "epic_prefix_tokens": args.epic_prefix_tokens,
        "start_layer": args.start_layer,
        "det_layer": args.det_layer,
        "end_layer": args.end_layer,
        "tau_dev": args.tau_dev,
        "tau_inf": args.tau_inf,
        "suffix_tokens": args.suffix_tokens,
        "adapter": (
            f"static shared_block prefetched under source prefix; true batched {args.method} adapter"
            if args.execution_path == "static-batched"
            else "static shared_block prefetched under source prefix; official RelayCaching placeholder path"
        ),
    }
    (run_config_path).write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    existing_rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str, str]] = set()
    if result_path.exists() and not args.overwrite:
        with result_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                existing_rows.append(row)
                # Failed rows have no ``correct`` key and are retried on resume.
                if "correct" in row:
                    completed_keys.add((row["task_id"], row["source_side"], row["target_side"]))
        pending_examples = [
            example
            for example in examples
            if (example.task_id, example.source, example.target) not in completed_keys
        ]
        print(
            f"Resuming {result_path}: {len(completed_keys)} successful directions already present; "
            f"{len(pending_examples)} remaining.",
            flush=True,
        )
    else:
        pending_examples = examples
    metadata_path = output_dir / "adapter_prompts.jsonl"
    write_metadata(metadata_path, records, explicit_reasoning=args.explicit_reasoning)

    if not pending_examples:
        successes = [row for row in existing_rows if "correct" in row]
        summary = {
            "method": args.method, "model": args.model,
            "input": str(args.input), "records": len(records), "evaluated_directions": len(examples),
            "use_debug_selection_direction": args.use_debug_selection_direction,
            "completed": len(successes), "failures": len(existing_rows) - len(successes),
            "correct": sum(bool(row["correct"]) for row in successes),
            "accuracy": (sum(bool(row["correct"]) for row in successes) / len(successes)) if successes else None,
            "explicit_reasoning": args.explicit_reasoning,
            "adapter": "static shared_block prefetched under source prefix; official RelayCaching placeholder path",
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    config = _make_config(args)
    model_path = MODEL_PATHS[args.model]
    print(f"[setup] model path: {model_path}", flush=True)
    print("[setup] loading model weights", flush=True)
    # Two lightweight facades share one loaded model and RelayCaching's official
    # shared-memory store, while keeping source/target node identities separate.
    source = LLMChat(model_path, config=config)
    target = LLMChat(model_path, config=config)
    print("[setup] model weights loaded", flush=True)
    _install_project_qwen_template(source, assistant_prefill=None)
    _install_project_qwen_template(
        target, assistant_prefill=None if args.explicit_reasoning else BOXED_PREFILL
    )

    correct = sum(bool(row["correct"]) for row in existing_rows if "correct" in row)
    completed = sum("correct" in row for row in existing_rows)
    failures = sum("correct" not in row for row in existing_rows)
    batch_inference_seconds = 0.0
    phase_totals: dict[str, float] = {}
    serial_online_ttft_seconds: list[float] = []
    serial_target_prefix_seconds: list[float] = []
    serial_generation_ttft_seconds: list[float] = []
    output_mode = "a" if existing_rows else "w"
    with result_path.open(output_mode, encoding="utf-8") as output:
        if args.execution_path == "static-batched":
            requested_direction_batch_size = args.batch_size * len(args.directions)
            effective_batch_size = requested_direction_batch_size
            scheduled_examples = pending_examples
            if args.method == "cacheblend" and args.cacheblend_bucket_by_length:
                # Length buckets reduce left-padding and make SDPA's batched
                # numerics closer to the single-item reference. This changes
                # only scheduling; result rows retain their task/direction IDs.
                scheduled_examples = sorted(
                    pending_examples,
                    key=lambda example: len(example.target_prefix) + len(example.shared_block)
                    + len(example.question),
                )
                print("[setup] cacheblend length bucketing enabled", flush=True)
            if args.warmup_batches:
                warmup_examples = scheduled_examples[:effective_batch_size]
                for _ in range(args.warmup_batches):
                    torch.cuda.synchronize()
                    _run_static_relay_batch(source, target, warmup_examples, args)
                    torch.cuda.synchronize()
                print(f"[setup] completed {args.warmup_batches} untimed warm-up batch(es)", flush=True)
            for start in range(0, len(scheduled_examples), effective_batch_size):
                batch_examples = scheduled_examples[start:start + effective_batch_size]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    batch_rows, phase_seconds = _run_static_relay_batch(source, target, batch_examples, args)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    batch_inference_seconds += elapsed
                    for name, seconds in phase_seconds.items():
                        phase_totals[name] = phase_totals.get(name, 0.0) + seconds
                    for example, row in zip(batch_examples, batch_rows):
                        row["elapsed_s"] = elapsed
                        correct += int(row["correct"])
                        completed += 1
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(
                            f"  output={row['output']!r} completed={row['output_with_prefill']!r} "
                            f"prediction={row['prediction']!r} gold={example.gold!r}",
                            flush=True,
                        )
                except Exception as error:
                    elapsed = time.perf_counter() - started
                    failures += len(batch_examples)
                    for example in batch_examples:
                        row = {
                            **example.as_metadata(explicit_reasoning=args.explicit_reasoning),
                            "method": args.method,
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(), "elapsed_s": elapsed,
                        }
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"  BATCH ERROR: {type(error).__name__}: {error}", flush=True)
                output.flush()
                print(
                    f"[{min(start + len(batch_examples), len(scheduled_examples))}/{len(scheduled_examples)}] "
                    f"{args.method} batch_size={len(batch_examples)} completed={completed} failures={failures} "
                    f"accuracy={(correct / completed) if completed else float('nan'):.3f}",
                    flush=True,
                )
            # The serial RelayCaching placeholder machinery below is deliberately
            # not entered: all KV assembly and decoding above used batched tensors.
            summary = {
                "method": args.method, "model": args.model,
                "input": str(args.input), "records": len(records), "evaluated_directions": len(examples),
                "use_debug_selection_direction": args.use_debug_selection_direction,
                "completed": completed, "failures": failures, "correct": correct,
                "accuracy": (correct / completed) if completed else None,
                "explicit_reasoning": args.explicit_reasoning,
                "batch_size": args.batch_size,
                "requested_direction_batch_size": requested_direction_batch_size,
                "effective_batch_size": effective_batch_size,
                "elapsed_seconds": batch_inference_seconds,
                "phase_seconds": phase_totals,
                "unattributed_seconds": max(0.0, batch_inference_seconds - sum(phase_totals.values())),
                # The source-side full prefill creates donor KV before the
                # target request. Do not charge it to online reuse TTFT.
                "cache_build_seconds": phase_totals.get("source_prefill", 0.0),
                "online_total_seconds": batch_inference_seconds - phase_totals.get("source_prefill", 0.0),
                "end_to_end_first_token_seconds": sum(
                    seconds for name, seconds in phase_totals.items()
                    if name not in {"prompt_build", "decode"}
                ),
                "first_token_pipeline_seconds": sum(
                    seconds for name, seconds in phase_totals.items()
                    if name not in {"prompt_build", "decode", "source_prefill"}
                ),
                "online_ttft_seconds": sum(
                    seconds for name, seconds in phase_totals.items()
                    if name not in {"prompt_build", "decode", "source_prefill"}
                ),
                "adapter": f"static shared_block prefetched under source prefix; true batched {args.method} adapter",
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=2))
            return

        for index, example in enumerate(pending_examples, start=1):
            message = f"{example.task_id}::{example.source}_to_{example.target}"
            request_uid = f"static-block-{uuid.uuid4().hex}"
            source.set_id(example.source, "source")
            target.set_id(example.target, "target")
            started = time.perf_counter()
            try:
                block_cache, block_ids, boundaries, hidden_states = _slice_shared_block_cache(
                    source,
                    source_prefix=example.source_prefix,
                    shared_block=example.shared_block,
                )
                _store_static_agent_block(
                    source, message=message, cache=block_cache, ids=block_ids,
                    hidden_states=hidden_states, hidden_layer=args.start_layer,
                )
                # Source-side block materialisation is the offline donor-cache
                # build. The target request starts here: its prefix prefill,
                # placeholder KV merge/relocation, correction, and first token
                # must all be included in online TTFT.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                online_started = time.perf_counter()
                await target.prepare_prefix_kv_segments(
                    target.node_id, SYSTEM_MESSAGE,
                    example.target_user_template(explicit_reasoning=args.explicit_reasoning),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prefix_elapsed = time.perf_counter() - online_started
                # Call the official core directly.  The public wrapper is
                # tenacity-decorated and otherwise hides the useful original
                # exception inside ``RetryError`` after its single attempt.
                result = await target.agen_generate(
                    messages=message,
                    max_tokens=args.max_new_tokens,
                    request_uid=request_uid,
                    mode="kv_reuse",
                    agent_id=target.node_id,
                    agent_name="target",
                    agent_role="target",
                    # ``qwen3-1.7b`` itself has a suffix, which RelayCaching's
                    # helper treats as a file path.  Pass the latency filename
                    # explicitly rather than the result directory.
                    output_dir=output_dir / "Latency.json",
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                online_elapsed = time.perf_counter() - online_started
                generation_ttft = float(result.metadata.get("generation_ttft", result.ttft))
                serial_online_ttft_seconds.append(online_elapsed)
                serial_target_prefix_seconds.append(prefix_elapsed)
                serial_generation_ttft_seconds.append(generation_ttft)
                # Generation continues an unfinished assistant message.  The
                # returned text is consequently only e.g. ``B}``, while answer
                # parsing must see the complete ``\\boxed{B}`` form.
                completed_text = result.text if args.explicit_reasoning else BOXED_PREFILL + result.text
                prediction = parse_prediction(example.dataset, completed_text)
                is_correct = prediction == str(example.gold).strip().upper()
                correct += int(is_correct)
                completed += 1
                row: dict[str, Any] = {
                    **example.as_metadata(explicit_reasoning=args.explicit_reasoning),
                    "method": args.method,
                    "prediction": prediction,
                    "correct": is_correct,
                    "output": result.text,
                    "output_with_prefill": completed_text,
                    "elapsed_s": time.perf_counter() - started,
                    "online_ttft_seconds": online_elapsed,
                    "target_prefix_prepare_seconds": prefix_elapsed,
                    "relay_metadata": result.metadata,
                    **boundaries,
                }
                print(
                    f"  output={result.text!r} completed={completed_text!r} "
                    f"prediction={prediction!r} gold={example.gold!r}",
                    flush=True,
                )
            except Exception as error:
                failures += 1
                row = {
                    **example.as_metadata(explicit_reasoning=args.explicit_reasoning), "method": args.method,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "elapsed_s": time.perf_counter() - started,
                }
                print(f"  ERROR: {row['error']}", flush=True)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            LLMChat.cleanup_message_kv_cache(message)
            print(
                f"[{index}/{len(pending_examples)}] {example.dataset} {example.source}_to_{example.target} "
                f"completed={completed} failures={failures} "
                f"accuracy={(correct / completed) if completed else float('nan'):.3f}",
                flush=True,
            )
    summary = {
        "method": args.method, "model": args.model,
        "input": str(args.input), "records": len(records), "evaluated_directions": len(examples),
        "use_debug_selection_direction": args.use_debug_selection_direction,
        "completed": completed, "failures": failures,
        "correct": correct,
        "accuracy": (correct / completed) if completed else None,
        "explicit_reasoning": args.explicit_reasoning,
        "execution_path": args.execution_path,
        "online_ttft_seconds": statistics.fmean(serial_online_ttft_seconds) if serial_online_ttft_seconds else None,
        "online_total_seconds": statistics.fmean(serial_online_ttft_seconds) if serial_online_ttft_seconds else None,
        "cache_build_seconds": 0.0,
        "target_prefix_prepare_seconds": (
            statistics.fmean(serial_target_prefix_seconds) if serial_target_prefix_seconds else None
        ),
        "generation_ttft_seconds": (
            statistics.fmean(serial_generation_ttft_seconds) if serial_generation_ttft_seconds else None
        ),
        "latency_samples": len(serial_online_ttft_seconds),
        "ttft_definition": (
            "target-prefix preparation plus official placeholder KV merge/relocation, "
            "selective correction, and first token; source donor-cache construction excluded"
        ),
        "adapter": "static shared_block prefetched under source prefix; official RelayCaching placeholder path",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> int:
    args = parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
