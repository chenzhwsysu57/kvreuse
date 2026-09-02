#!/usr/bin/env python3
"""Run paired Qwen3 direct shared-block KV reuse with RoPE relocation.

For every dataset record this runner computes full-A/full-B once, then executes
the same cache-transplant path for A->A, A->B, B->A, and B->B.  A->A and B->B
are directional implementation controls.  Cached post-RoPE Keys are moved from
their source absolute positions to the target positions; Values are not rotated.
``--method clean_reuse`` adds a control in which the shared block is encoded
without any prefix before being transplanted into each target prefix.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


SYSTEM_MESSAGE = "You're a helpful assistant."
MODEL_IDS = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
    "8b": "Qwen/Qwen3-8B",
}
SELF_CONDITIONS = (("a_to_a", "a", "a"), ("b_to_b", "b", "b"))
CROSS_CONDITIONS = (("a_to_b", "a", "b"), ("b_to_a", "b", "a"))
ALL_CONDITIONS = (
    ("a_to_a", "a", "a"),
    ("a_to_b", "a", "b"),
    ("b_to_a", "b", "a"),
    ("b_to_b", "b", "b"),
)


@dataclass
class PromptParts:
    rendered: str
    full_ids: torch.Tensor
    prefix_ids: torch.Tensor
    block_ids: torch.Tensor
    suffix_ids: torch.Tensor
    block_token_start: int
    block_token_end: int
    block_char_start: int
    block_char_end: int
    cached_char_start: int
    cached_char_end: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def sample_hit_limit(row: dict[str, Any]) -> bool:
    return any(
        value.get("hit_max_new_tokens", False)
        for family in ("full", "reuse")
        for value in row.get(family, {}).values()
    )


def prepare_truncation_retries(
    predictions_path: Path, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], set[str]]:
    truncated = [row for row in rows if sample_hit_limit(row)]
    retry_ids = {row["task_id"] for row in truncated}
    if not truncated:
        return rows, retry_ids
    archive_path = predictions_path.with_name("truncated_samples.jsonl")
    archived = {
        (row["task_id"], max(
            value.get("generation_max_new_tokens", value.get("output_tokens", 0))
            for family in ("full", "reuse") for value in row.get(family, {}).values()
        ))
        for row in load_jsonl(archive_path)
    } if archive_path.exists() else set()
    with archive_path.open("a", encoding="utf-8") as handle:
        for row in truncated:
            limit = max(
                value.get("generation_max_new_tokens", value.get("output_tokens", 0))
                for family in ("full", "reuse") for value in row.get(family, {}).values()
            )
            if (row["task_id"], limit) not in archived:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    retained = [row for row in rows if row["task_id"] not in retry_ids]
    write_jsonl_atomic(predictions_path, retained)
    return retained, retry_ids


def parse_prediction(dataset: str, output: str) -> str | None:
    text = output.strip()
    boxed = re.findall(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", text, re.I)
    if boxed:
        text = boxed[-1].strip()
        if dataset == "harmbench_contextual":
            return text.upper() if text.upper() in {"BENIGN", "HARMFUL"} else None
        if dataset == "helpsteer2":
            return text if text in {"0", "4"} else None
        match = re.fullmatch(r"[A-H]", text.upper())
        return match.group(0) if match else None
    if dataset == "harmbench_contextual":
        match = re.match(r"^\s*(?:intent\s*[:=]\s*)?(BENIGN|HARMFUL)\b", text, re.I)
        if match:
            return match.group(1).upper()
        explicit = re.findall(
            r"(?:intent|classification|answer)\s*(?:is|:|=)\s*(BENIGN|HARMFUL)\b", text, re.I
        )
        return explicit[-1].upper() if explicit else None
    if dataset == "helpsteer2":
        match = re.match(r"^\s*(?:score\s*[:=]\s*)?([04])(?!\d)", text, re.I)
        if match:
            return match.group(1)
        explicit = re.findall(r"(?:score|rating|answer)\s*(?:is|:|=)\s*([04])(?!\d)", text, re.I)
        return explicit[-1] if explicit else None
    upper = text.upper()
    start = re.match(r"^\s*\[?([A-H])\]?(?:\s|$|[.)])", upper)
    if start:
        return start.group(1)
    explicit = re.findall(
        r"(?:FINAL(?:\s+ANSWER)?|ANSWER|OPTION|ALLOCATION|CHOICE)\s*(?:IS|:|=)\s*\[?([A-H])\]?",
        upper,
    )
    if explicit:
        return explicit[-1]
    end = re.search(r"(?:^|\n)\s*\[?([A-H])\]?[.!]?\s*$", upper)
    return end.group(1) if end else None


def build_prompt_parts(
    tokenizer: Any, record: dict[str, Any], side: str, *, enable_thinking: bool,
    explicit_reasoning: bool, boxed_output: bool,
    newline_position: str = "none", newline_count: int = 0,
) -> PromptParts:
    prefix = record[f"prefix_{side}"]
    block = record["shared_block"]
    if newline_position == "prefix_block":
        prefix_block_separator = "\n" * max(1, newline_count)
    else:
        prefix_block_separator = "\n\n"
    user_message = f"{prefix}{prefix_block_separator}{block}\n\n{record['question']}"
    if newline_position == "end" and newline_count > 0:
        user_message += "\n" * newline_count
    if explicit_reasoning:
        user_message += (
            "\n\nFirst reason about the task step by step. Then put your final answer "
            "in exactly one \\boxed{...}."
        )
    assistant_prefill = None
    if boxed_output:
        if explicit_reasoning:
            # The explicit-reasoning instruction already specifies the final
            # boxed-answer format; do not duplicate it.
            pass
        else:
            user_message += (
                "\n\nDo not explain or reason. Output only the final answer in exactly one "
                "\\boxed{...}."
            )
            assistant_prefill = "The answer is: \\boxed{"
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message},
    ]
    if assistant_prefill is not None:
        messages.append({"role": "assistant", "content": assistant_prefill})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=assistant_prefill is None,
        continue_final_message=assistant_prefill is not None,
        enable_thinking=enable_thinking,
    )
    user_start = rendered.find(user_message)
    if user_start < 0:
        raise ValueError("chat template did not preserve the user message verbatim")
    block_start = user_start + len(prefix) + len(prefix_block_separator)
    block_end = block_start + len(block)
    if rendered[block_start:block_end] != block:
        raise ValueError("failed to locate the exact shared block in the rendered prompt")

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    # BPE tokens can straddle a natural-language boundary (for example, a
    # period plus the following newlines).  A cache cannot contain half a
    # token, so use the maximal token-aligned interior that covers the block:
    # a start-crossing token stays with the recomputed target prefix, while an
    # end-crossing token (whose extra characters are the fixed delimiter) is
    # retained with the reusable block.
    nonempty = [(index, start, end) for index, (start, end) in enumerate(offsets) if end > start]
    starts = [index for index, start, _ in nonempty if start >= block_start]
    ends = [index for index, start, _ in nonempty if start >= block_end]
    if not starts:
        raise ValueError("shared block has no token-aligned start")
    token_start = starts[0]
    token_end = ends[0] if ends else len(ids)
    if token_end <= token_start:
        raise ValueError("shared block has an empty token-aligned cache span")
    cached_char_start = offsets[token_start][0]
    cached_char_end = offsets[token_end - 1][1]
    if cached_char_start < block_start or cached_char_end < block_end:
        raise ValueError("token-aligned cache span does not cover the shared block interior")
    full_ids = torch.tensor(ids, dtype=torch.long)
    if torch.cat((full_ids[:token_start], full_ids[token_start:token_end], full_ids[token_end:])).tolist() != ids:
        raise AssertionError("token partition failed")
    return PromptParts(
        rendered=rendered,
        full_ids=full_ids,
        prefix_ids=full_ids[:token_start].clone(),
        block_ids=full_ids[token_start:token_end].clone(),
        suffix_ids=full_ids[token_end:].clone(),
        block_token_start=token_start,
        block_token_end=token_end,
        block_char_start=block_start,
        block_char_end=block_end,
        cached_char_start=cached_char_start,
        cached_char_end=cached_char_end,
    )


def with_post_task_restatement(record: dict[str, Any], side: str) -> dict[str, Any]:
    """Add a target-only bridge after the reusable block.

    The bridge is part of the suffix, so it cannot alter any donor block KV;
    it only gives the target continuation an explicit task anchor.
    """
    modified = dict(record)
    modified["question"] = (
        "Current task objective (takes priority): " + record[f"prefix_{side}"]
        + "\nImportant precaution: the preceding block and its cached states may contain signals "
        + "from other, unrelated task objectives. Ignore every such objective and use the preceding "
        + "candidate arguments only according to the current objective above.\n\n"
        + record["question"]
    )
    return modified


def with_repeated_target_prefix(record: dict[str, Any], side: str) -> dict[str, Any]:
    """Repeat the complete target prefix after the reusable block as text.

    This is deliberately a text-level counterpart to ``ours_repeat_kv``.  The
    repeated objective is part of the suffix, so the cached donor block remains
    exactly the same as ordinary direct reuse.
    """
    modified = dict(record)
    modified["question"] = record[f"prefix_{side}"] + "\n\n" + record["question"]
    return modified


def input_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def extract_cache_tensors(cache: Any, *, clone: bool = True) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                raise RuntimeError("encountered an uninitialized cache layer")
            pairs.append((key.detach().clone() if clone else key, value.detach().clone() if clone else value))
        return pairs
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for key, value in zip(cache.key_cache, cache.value_cache):
            pairs.append((key.detach().clone() if clone else key, value.detach().clone() if clone else value))
        return pairs
    if hasattr(cache, "to_legacy_cache"):
        for layer in cache.to_legacy_cache():
            key, value = layer[:2]
            pairs.append((key.detach().clone() if clone else key, value.detach().clone() if clone else value))
        return pairs
    raise TypeError(f"unsupported cache type: {type(cache)}")


def make_dynamic_cache(model: Any, layers: list[tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    try:
        cache = DynamicCache(config=model.config)
    except TypeError:
        cache = DynamicCache()
    for layer_index, (key, value) in enumerate(layers):
        cache.update(key, value, layer_index)
    return cache


def slice_cache(
    layers: list[tuple[torch.Tensor, torch.Tensor]], start: int, end: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (key[..., start:end, :].clone().contiguous(), value[..., start:end, :].clone().contiguous())
        for key, value in layers
    ]


def rotate_half_qwen(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    half = tensor.shape[-1] // 2
    return torch.cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


def assert_default_rope(model: Any) -> None:
    scaling = getattr(model.config, "rope_scaling", None)
    parameters = getattr(model.config, "rope_parameters", None)
    candidate = scaling if scaling is not None else parameters
    if isinstance(candidate, dict):
        rope_type = candidate.get("rope_type", candidate.get("type", "default"))
        if rope_type not in (None, "default"):
            raise ValueError(f"direct reuse currently supports default RoPE only, got {rope_type!r}")


def rope_cos_sin(
    model: Any, reference_key: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rotary is None:
        raise RuntimeError("Qwen3 decoder does not expose model.rotary_emb")
    positions = positions.to(device=reference_key.device, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(
        (1, positions.shape[1], reference_key.shape[-1]),
        device=reference_key.device,
        dtype=reference_key.dtype,
    )
    with torch.inference_mode():
        cos, sin = rotary(dummy, positions)
    return cos.unsqueeze(1).float(), sin.unsqueeze(1).float()


def relocate_post_rope_key(
    model: Any, key: torch.Tensor, source_start: int, target_start: int
) -> torch.Tensor:
    """Undo source-position RoPE and apply target-position RoPE."""
    if source_start == target_start:
        return key.clone()
    length = key.shape[-2]
    source_positions = torch.arange(source_start, source_start + length, device=key.device)
    target_positions = torch.arange(target_start, target_start + length, device=key.device)
    cos, sin = rope_cos_sin(model, key, torch.cat((source_positions, target_positions)))
    cos_source, cos_target = cos[..., :length, :], cos[..., length:, :]
    sin_source, sin_target = sin[..., :length, :], sin[..., length:, :]
    rotated = key.float()
    unrotated = rotated * cos_source - rotate_half_qwen(rotated) * sin_source
    relocated = unrotated * cos_target + rotate_half_qwen(unrotated) * sin_target
    return relocated.to(dtype=key.dtype).contiguous()


def relocate_block(
    model: Any,
    block: list[tuple[torch.Tensor, torch.Tensor]],
    source_start: int,
    target_start: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (relocate_post_rope_key(model, key, source_start, target_start), value.clone().contiguous())
        for key, value in block
    ]


def splice_prefix_block(
    prefix: list[tuple[torch.Tensor, torch.Tensor]],
    block: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if len(prefix) != len(block):
        raise ValueError("prefix and block caches have different layer counts")
    return [
        (
            torch.cat((prefix_key, block_key), dim=-2).contiguous(),
            torch.cat((prefix_value, block_value), dim=-2).contiguous(),
        )
        for (prefix_key, prefix_value), (block_key, block_value) in zip(prefix, block)
    ]


@torch.inference_mode()
def forward_ids(model: Any, ids: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    device = input_device(model)
    input_ids = ids.unsqueeze(0).to(device)
    length = input_ids.shape[1]
    positions = torch.arange(length, device=device, dtype=torch.long)
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones((1, length), device=device, dtype=torch.long),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        use_cache=True,
        return_dict=True,
    )
    logits = output.logits[0, -1].detach().float()
    layers = extract_cache_tensors(output.past_key_values)
    del output, input_ids, positions
    return logits, layers


@torch.inference_mode()
def forward_suffix(
    model: Any,
    past_layers: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    if suffix_ids.numel() == 0:
        raise ValueError("the rendered prompt has an empty suffix")
    device = input_device(model)
    past_length = past_layers[0][0].shape[-2]
    suffix = suffix_ids.unsqueeze(0).to(device)
    positions = torch.arange(past_length, past_length + suffix.shape[1], device=device, dtype=torch.long)
    cache = make_dynamic_cache(model, past_layers)
    output = model(
        input_ids=suffix,
        attention_mask=torch.ones((1, past_length + suffix.shape[1]), device=device, dtype=torch.long),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    logits = output.logits[0, -1].detach().float()
    layers = extract_cache_tensors(output.past_key_values)
    del output, suffix, positions, cache
    return logits, layers


@torch.inference_mode()
def greedy_continue(
    model: Any,
    tokenizer: Any,
    initial_logits: torch.Tensor,
    prompt_layers: list[tuple[torch.Tensor, torch.Tensor]],
    max_new_tokens: int,
) -> tuple[list[int], bool]:
    device = input_device(model)
    cache = make_dynamic_cache(model, prompt_layers)
    past_length = prompt_layers[0][0].shape[-2]
    current = initial_logits.argmax().view(1, 1).to(device)
    generated = [int(current.item())]
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    if generated[-1] in eos_ids:
        return generated, False
    for _ in range(max_new_tokens - 1):
        position = torch.tensor([past_length], device=device, dtype=torch.long)
        output = model(
            input_ids=current,
            attention_mask=torch.ones((1, past_length + 1), device=device, dtype=torch.long),
            position_ids=position.unsqueeze(0),
            cache_position=position,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        current = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(current.item()))
        past_length += 1
        if generated[-1] in eos_ids:
            del output
            return generated, False
        del output
    return generated, generated[-1] not in eos_ids


def logits_comparison(full_logits: torch.Tensor, reuse_logits: torch.Tensor) -> dict[str, float]:
    full_logp = torch.log_softmax(full_logits.float(), dim=-1)
    reuse_logp = torch.log_softmax(reuse_logits.float(), dim=-1)
    kl = torch.sum(full_logp.exp() * (full_logp - reuse_logp))
    cosine = torch.nn.functional.cosine_similarity(full_logits.float(), reuse_logits.float(), dim=0)
    return {
        "first_token_kl_full_to_reuse": float(kl.item()),
        "first_token_logit_cosine": float(cosine.item()),
        "first_token_max_abs_logit_diff": float((full_logits - reuse_logits).abs().max().item()),
    }


def layer_token_cosines(
    reused: list[tuple[torch.Tensor, torch.Tensor]],
    target: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[np.ndarray, np.ndarray]:
    key_columns, value_columns = [], []
    for (reuse_key, reuse_value), (target_key, target_value) in zip(reused, target):
        key = torch.nn.functional.cosine_similarity(reuse_key.float(), target_key.float(), dim=-1)
        value = torch.nn.functional.cosine_similarity(reuse_value.float(), target_value.float(), dim=-1)
        key_columns.append(key.mean(dim=(0, 1)).cpu().numpy())
        value_columns.append(value.mean(dim=(0, 1)).cpu().numpy())
    return np.stack(key_columns, axis=1), np.stack(value_columns, axis=1)


def safe_task_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:180]


def save_similarity_artifacts(
    sample_dir: Path,
    matrices: dict[str, dict[str, np.ndarray]],
) -> tuple[Path, Path]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    npz_path = sample_dir / "kv_similarity.npz"
    arrays = {
        f"{condition}_{metric}": value
        for condition, values in matrices.items()
        for metric, value in values.items()
    }
    np.savez_compressed(npz_path, **arrays)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered_conditions = [name for name, _, _ in ALL_CONDITIONS if name in matrices]
    figure, axes = plt.subplots(
        len(ordered_conditions),
        2,
        figsize=(13, 3.5 * len(ordered_conditions)),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row_index, condition in enumerate(ordered_conditions):
        for column_index, metric in enumerate(("key_rope_corrected", "value")):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                matrices[condition][metric],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                # Match the reference figures: low similarity is dark purple,
                # high similarity is yellow.  Cosine values below zero are
                # clipped at the 0 endpoint so that 0 has a stable color.
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_title(f"{condition}: {'Key (RoPE corrected)' if column_index == 0 else 'Value'}")
            axis.set_xlabel("Layer index")
            axis.set_ylabel("Shared-block token index")
    if image is not None:
        figure.colorbar(image, ax=axes, label="Cosine similarity", shrink=0.72)
    plot_path = sample_dir / "kv_similarity.png"
    figure.savefig(plot_path, dpi=170)
    plt.close(figure)
    return npz_path, plot_path


def result_from_generation(
    tokenizer: Any,
    dataset: str,
    gold: str,
    token_ids: list[int],
    hit_limit: bool,
    elapsed: float,
    generation_limit: int,
    enable_thinking: bool,
    explicit_reasoning: bool = False,
    output_prefix: str = "",
) -> dict[str, Any]:
    generated_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    text = f"{output_prefix}{generated_text}" if output_prefix else generated_text
    reasoning_text = ""
    answer_text = text
    if enable_thinking:
        marker = "</think>"
        if marker in text:
            reasoning_text, answer_text = text.rsplit(marker, 1)
            reasoning_text = reasoning_text.strip()
            answer_text = answer_text.strip()
    elif explicit_reasoning:
        marker = re.search(r"\\boxed\s*\{", text, re.I)
        if marker:
            reasoning_text = text[: marker.start()].strip()
            answer_text = text[marker.start() :].strip()
    prediction = parse_prediction(dataset, answer_text)
    return {
        "output_text": text,
        "reasoning_text": reasoning_text,
        "answer_text": answer_text,
        "output_token_ids": token_ids,
        "output_tokens": len(token_ids),
        "prediction": prediction,
        "gold": gold,
        "correct": prediction == gold,
        "hit_max_new_tokens": hit_limit,
        "generation_max_new_tokens": generation_limit,
        "elapsed_seconds": elapsed,
    }


def summarize(rows: list[dict[str, Any]], *, include_datasets: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {"completed_samples": len(rows), "full": {}, "reuse": {}}
    if not rows:
        return summary
    full_rows = [row for row in rows if row.get("full")]
    reuse_rows = [row for row in rows if row.get("reuse")]
    for side in ("a", "b"):
        values = [row["full"][side] for row in full_rows]
        if not values:
            continue
        summary["full"][side] = {
            "accuracy": sum(value["correct"] for value in values) / len(values),
            "parse_rate": sum(value["prediction"] is not None for value in values) / len(values),
            "hit_max_new_tokens": sum(value["hit_max_new_tokens"] for value in values),
            "mean_elapsed_seconds": sum(value["elapsed_seconds"] for value in values) / len(values),
        }
    both_full = [row["full"][side] for row in full_rows for side in ("a", "b")]
    if both_full:
        summary["full"]["both"] = {
            "outputs": len(both_full),
            "accuracy": sum(value["correct"] for value in both_full) / len(both_full),
            "parse_rate": sum(value["prediction"] is not None for value in both_full) / len(both_full),
            "hit_max_new_tokens": sum(value["hit_max_new_tokens"] for value in both_full),
        }
    observed_conditions = [
        name for name, _, _ in ALL_CONDITIONS
        if any(name in row.get("reuse", {}) for row in reuse_rows)
    ]
    targets = {name: target for name, _, target in ALL_CONDITIONS}
    sources = {name: source for name, source, _ in ALL_CONDITIONS}
    for condition in observed_conditions:
        target = targets[condition]
        source = sources[condition]
        paired = [
            (row.get("full", {}).get(source), row.get("full", {}).get(target), row["reuse"][condition])
            for row in reuse_rows if condition in row["reuse"]
        ]
        source_values = [source_full for source_full, _, _ in paired if source_full is not None]
        full_values = [target_full for _, target_full, _ in paired if target_full is not None]
        values = [value for _, _, value in paired]
        metrics = {
            "source_side": source,
            "target_side": target,
            "accuracy": sum(value["correct"] for value in values) / len(values),
            "parse_rate": sum(value["prediction"] is not None for value in values) / len(values),
            "hit_max_new_tokens": sum(value["hit_max_new_tokens"] for value in values),
            "mean_first_token_kl": sum(value["first_token_kl_full_to_reuse"] for value in values) / len(values),
            "mean_key_cosine_before_rope": sum(
                value["mean_key_cosine_before_rope"] for value in values
            ) / len(values),
            "mean_key_cosine": sum(value["mean_key_cosine"] for value in values) / len(values),
            "mean_key_cosine_rope_gain": sum(
                value["mean_key_cosine"] - value["mean_key_cosine_before_rope"] for value in values
            ) / len(values),
            "mean_value_cosine": sum(value["mean_value_cosine"] for value in values) / len(values),
            "mean_elapsed_seconds": sum(value["elapsed_seconds"] for value in values) / len(values),
        }
        if full_values:
            metrics.update({
                "paired_accuracy_loss_full_minus_reuse": sum(int(full["correct"]) - int(value["correct"]) for full, value in zip(full_values, values)) / len(values),
                "prediction_agreement_with_full": sum(full["prediction"] == value["prediction"] for full, value in zip(full_values, values)) / len(values),
                "prediction_agreement_with_source_full": sum(full["prediction"] == value["prediction"] for full, value in zip(source_values, values)) / len(values),
                "full_correct_reuse_wrong": sum(full["correct"] and not value["correct"] for full, value in zip(full_values, values)),
                "full_wrong_reuse_correct": sum(not full["correct"] and value["correct"] for full, value in zip(full_values, values)),
            })
        summary["reuse"][condition] = metrics
    cross_pairs = [
        (row["full"][target], row["reuse"][condition])
        for row in full_rows
        if row.get("reuse")
        for condition, target in (("a_to_b", "b"), ("b_to_a", "a"))
    ]
    if cross_pairs:
        summary["cross_direction_aggregate"] = {
        "outputs": len(cross_pairs),
        "full_target_accuracy": sum(full["correct"] for full, _ in cross_pairs) / len(cross_pairs),
        "direct_reuse_accuracy": sum(reuse["correct"] for _, reuse in cross_pairs) / len(cross_pairs),
        "direct_reuse_parse_rate": sum(
            reuse["prediction"] is not None for _, reuse in cross_pairs
        ) / len(cross_pairs),
        "hit_max_new_tokens": sum(reuse["hit_max_new_tokens"] for _, reuse in cross_pairs),
        "paired_accuracy_loss_full_minus_reuse": sum(
            int(full["correct"]) - int(reuse["correct"]) for full, reuse in cross_pairs
        ) / len(cross_pairs),
        "prediction_agreement_with_full": sum(
            full["prediction"] == reuse["prediction"] for full, reuse in cross_pairs
        ) / len(cross_pairs),
        "mean_first_token_kl": sum(
            reuse["first_token_kl_full_to_reuse"] for _, reuse in cross_pairs
        ) / len(cross_pairs),
        }
    self_values = [
        control["pass"]
        for row in rows
        for control in row.get("self_controls", {}).values()
    ]
    summary["self_controls_run"] = len(self_values)
    summary["self_controls_pass"] = all(self_values) if self_values else None
    if include_datasets:
        datasets = sorted({row["dataset"] for row in rows})
        summary["by_dataset"] = {
            dataset: summarize(
                [row for row in rows if row["dataset"] == dataset], include_datasets=False
            )
            for dataset in datasets
        }
    return summary


def validate_model_files(model_path: Path) -> None:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        missing = sorted({name for name in index["weight_map"].values() if not (model_path / name).is_file()})
        if missing:
            raise FileNotFoundError(f"model cache is incomplete; missing {missing}")
    incomplete = list(model_path.glob("*.incomplete"))
    if incomplete:
        raise FileNotFoundError(f"model cache contains incomplete files: {incomplete}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results/direct_reuse"))
    parser.add_argument(
        "--method", choices=(
            "all", "full", "reuse", "clean_reuse", "tail16_recompute", "tail16_post_recompute",
            "ours_repeat_txt", "ours_repeat_kv",
        ), default="all",
        help="Generate dense full outputs, source-prefixed RoPE cross-reuse, or clean-block RoPE reuse. "
             "clean_reuse encodes the shared block with no prefix at all. tail16_recompute reuses the "
             "source-conditioned block except for its final 16 tokens, which are recomputed under the "
             "target cache. tail16_post_recompute additionally inserts a target-task restatement after "
             "the block. ours_repeat_txt repeats the complete target prefix after the block as text; "
             "ours_repeat_kv appends that target-prefix cache after the transplanted block with RoPE "
             "relocation. Reuse-only modes still perform "
             "non-generative target full forwards for reference logits and KV similarity.",
    )
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--task-ids", nargs="*")
    parser.add_argument("--max-samples", type=int, default=0, help="0 keeps every selected sample")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retry-truncated", action="store_true")
    parser.add_argument("--retry-max-new-tokens", type=int, default=1024)
    parser.add_argument("--self-kl-atol", type=float, default=1e-2)
    parser.add_argument("--self-logit-cos-min", type=float, default=0.999)
    parser.add_argument(
        "--run-self-controls",
        action="store_true",
        help="also run A->A and B->B; normally needed only after cache/runtime changes",
    )
    parser.add_argument("--fail-on-self-mismatch", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable Qwen3 thinking mode; score only the answer after </think>",
    )
    parser.add_argument("--explicit-reasoning", action="store_true",
                        help="ask for brief visible reasoning (not Qwen3 thinking mode)")
    parser.add_argument("--boxed-output", action="store_true",
                        help="require the final answer inside exactly one \\boxed{...}")
    parser.add_argument("--placeholder-tokens", type=int, default=0,
                        help="append this many fixed placeholder tokens before generation")
    parser.add_argument("--placeholder-text", default="\n",
                        help="text tokenized and repeated for placeholder-tokens")
    parser.add_argument("--newline-position", choices=("none", "prefix_block", "end"), default="none",
                        help="insert fixed newlines in the prompt at this location")
    parser.add_argument("--newline-count", type=int, default=0,
                        help="number of newlines for --newline-position")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.retry_truncated and args.retry_max_new_tokens <= args.max_new_tokens:
        raise ValueError("--retry-max-new-tokens must exceed --max-new-tokens")
    if args.run_self_controls and args.method != "all":
        raise ValueError("--run-self-controls requires --method all so it can compare against generated full outputs")

    records = load_jsonl(args.input)
    if args.datasets:
        records = [record for record in records if record["dataset"] in set(args.datasets)]
    if args.task_ids:
        records = [record for record in records if record["task_id"] in set(args.task_ids)]
    if args.max_samples > 0:
        records = records[:args.max_samples]
    if not records:
        raise ValueError("no records selected")

    model_id = MODEL_IDS[args.model]
    model_path = Path(snapshot_download(model_id, local_files_only=not args.allow_download))
    validate_model_files(model_path)
    output_dir = args.output_root / f"qwen3-{args.model}"
    sample_artifact_root = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    predictions_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "run_config.json"
    conditions = ALL_CONDITIONS if args.run_self_controls else CROSS_CONDITIONS
    run_full = args.method in {"all", "full"}
    run_reuse = args.method in {
        "all", "reuse", "clean_reuse", "tail16_recompute", "tail16_post_recompute",
        "ours_repeat_txt", "ours_repeat_kv",
    }
    clean_reuse = args.method == "clean_reuse"
    tail16_recompute = args.method in {"tail16_recompute", "tail16_post_recompute"}
    post_restatement = args.method == "tail16_post_recompute"
    repeat_prefix_text = args.method == "ours_repeat_txt"
    repeat_prefix_kv = args.method == "ours_repeat_kv"
    run_config = {
        "experiment": "direct_shared_block_kv_reuse",
        "method": args.method,
        "model_id": model_id,
        "model_path": str(model_path),
        "input_path": str(args.input),
        "input_sha256": file_sha256(args.input),
        "selected_task_ids": [record["task_id"] for record in records],
        "conditions": [condition for condition, _, _ in conditions],
        "run_self_controls": args.run_self_controls,
        "system_message": SYSTEM_MESSAGE,
        "chat_template": "tokenizer.apply_chat_template",
        "enable_thinking": args.enable_thinking,
        "explicit_reasoning": args.explicit_reasoning,
        "boxed_output": args.boxed_output,
        "placeholder_tokens": args.placeholder_tokens,
        "placeholder_text": args.placeholder_text,
        "newline_position": args.newline_position,
        "newline_count": args.newline_count,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "truncation_retry_max_new_tokens": args.retry_max_new_tokens,
        "truncation_retry_policy": "archive_and_rerun_complete_sample",
        "rope_correction": "inverse_source_then_apply_target_using_model.rotary_emb",
        "reuse_block_context": "no_prefix" if clean_reuse else "source_agent_prefix",
        "tail_recompute_tokens": 16 if tail16_recompute else 0,
        "post_task_restatement": post_restatement,
        "repeat_target_prefix_text": repeat_prefix_text,
        "repeat_target_prefix_kv": repeat_prefix_kv,
        "repeat_target_prefix_kv_rope": (
            "inverse_source_then_apply_target_at_prefix_plus_block" if repeat_prefix_kv else None
        ),
        "value_correction": "none",
        "self_kl_atol": args.self_kl_atol,
        "self_logit_cos_min": args.self_logit_cos_min,
        "torch_version": torch.__version__,
    }
    if args.overwrite:
        predictions_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)
    if config_path.exists():
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        migration_fields = {"truncation_retry_max_new_tokens", "truncation_retry_policy"}
        old_core = {key: old_config.get(key) for key in run_config if key not in migration_fields}
        new_core = {key: value for key, value in run_config.items() if key not in migration_fields}
        if old_core != new_core:
            raise ValueError("existing run_config differs; use --overwrite or another output root")
    write_json_atomic(config_path, run_config)
    existing = load_jsonl(predictions_path) if predictions_path.exists() else []
    retry_task_ids: set[str] = set()
    if args.retry_truncated:
        existing, retry_task_ids = prepare_truncation_retries(predictions_path, existing)
        if retry_task_ids:
            print(
                f"Retrying {len(retry_task_ids)} truncated sample(s) with "
                f"max_new_tokens={args.retry_max_new_tokens}",
                flush=True,
            )
    completed = {row["task_id"] for row in existing}
    if all(record["task_id"] in completed for record in records):
        final_summary = {"config": run_config, "metrics": summarize(existing)}
        write_json_atomic(summary_path, final_summary)
        print(f"All {len(records)} selected samples are already complete; no model load needed.")
        print(json.dumps(final_summary["metrics"], indent=2))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for incomplete direct-reuse runs")
    print(f"Loading {model_id} from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    assert_default_rope(model)
    placeholder_ids = None
    if args.placeholder_tokens > 0:
        base_placeholder = tokenizer(
            args.placeholder_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        if base_placeholder.numel() == 0:
            raise ValueError("--placeholder-text produced no tokens")
        repeats = (args.placeholder_tokens + base_placeholder.numel() - 1) // base_placeholder.numel()
        placeholder_ids = base_placeholder.repeat(repeats)[: args.placeholder_tokens]

    # Exclude one-time CUDA/kernel initialization from the first measured full
    # prefill.  The warm-up result is discarded and never enters evaluation.
    warmup_ids = tokenizer("warmup", add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    _, warmup_layers = forward_ids(model, warmup_ids)
    del warmup_layers
    sync_cuda()

    for sample_index, record in enumerate(records, 1):
        if record["task_id"] in completed:
            continue
        generation_limit = (
            args.retry_max_new_tokens if record["task_id"] in retry_task_ids else args.max_new_tokens
        )
        parts = {
            side: build_prompt_parts(
                tokenizer,
                with_post_task_restatement(record, side) if post_restatement else
                with_repeated_target_prefix(record, side) if repeat_prefix_text else record,
                side,
                                     enable_thinking=args.enable_thinking,
                                     explicit_reasoning=args.explicit_reasoning,
                                     boxed_output=args.boxed_output,
                                     newline_position=args.newline_position,
                                     newline_count=args.newline_count)
            for side in ("a", "b")
        }
        if not torch.equal(parts["a"].block_ids, parts["b"].block_ids):
            raise ValueError(f"{record['task_id']}: shared block token IDs differ across prefixes")

        full_results: dict[str, dict[str, Any]] = {}
        full_logits: dict[str, torch.Tensor] = {}
        full_blocks: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        target_prefixes: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        prefix_seconds: dict[str, float] = {}
        for side in ("a", "b"):
            sync_cuda(); started = time.perf_counter()
            logits, full_layers = forward_ids(model, parts[side].full_ids)
            if run_full:
                generation_layers = full_layers
                generation_logits = logits
                if placeholder_ids is not None:
                    generation_logits, generation_layers = forward_suffix(model, full_layers, placeholder_ids)
                tokens, hit_limit = greedy_continue(
                    model, tokenizer, generation_logits, generation_layers, generation_limit
                )
                sync_cuda(); elapsed = time.perf_counter() - started
                full_results[side] = result_from_generation(
                    tokenizer,
                    record["dataset"],
                    record[f"gold_{side}"],
                    tokens,
                    hit_limit,
                    elapsed,
                    generation_limit,
                    args.enable_thinking,
                    args.explicit_reasoning,
                    "The answer is: \\boxed{" if args.boxed_output and not args.explicit_reasoning else "",
                )
                if generation_layers is not full_layers:
                    del generation_layers
            full_logits[side] = logits

            if run_reuse:
                full_blocks[side] = slice_cache(
                    full_layers, parts[side].block_token_start, parts[side].block_token_end
                )
                sync_cuda(); prefix_started = time.perf_counter()
                _, target_prefixes[side] = forward_ids(model, parts[side].prefix_ids)
                sync_cuda(); prefix_seconds[side] = time.perf_counter() - prefix_started
            del full_layers

        # Clean control: encode precisely the already-tokenized reusable block
        # at positions 0..B-1, with no system message or agent prefix.  This
        # isolates the effect of source-prefix-conditioned KV from the effect
        # of reusing the block's lexical content itself.
        clean_block: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if clean_reuse:
            _, clean_layers = forward_ids(model, parts["a"].block_ids)
            clean_block = slice_cache(clean_layers, 0, parts["a"].block_ids.numel())
            del clean_layers

        matrices: dict[str, dict[str, np.ndarray]] = {}
        reuse_results: dict[str, dict[str, Any]] = {}
        for condition, source, target in (conditions if run_reuse else ()):
            source_start = 0 if clean_reuse else parts[source].block_token_start
            target_start = parts[target].block_token_start
            source_block = clean_block if clean_reuse else full_blocks[source]
            if source_block is None:
                raise AssertionError("clean block cache was not materialised")
            raw_key, _ = layer_token_cosines(source_block, full_blocks[target])
            sync_cuda(); started = time.perf_counter()
            relocated = relocate_block(model, source_block, source_start, target_start)
            recomputed_tail_tokens = 0
            if tail16_recompute:
                block_length = parts[target].block_ids.numel()
                # Short shared blocks occur in HelpSteer2.  Recompute the whole
                # block in that case rather than aborting the benchmark; for
                # longer blocks this preserves the usual final-16-token policy.
                recomputed_tail_tokens = min(16, block_length)
                donor_length = block_length - recomputed_tail_tokens
                # The donor prefix retains the relocated KV, while the final
                # min(16, B) tokens are run serially under the target cache.
                mixed = splice_prefix_block(
                    target_prefixes[target], slice_cache(relocated, 0, donor_length)
                )
                for block_index in range(donor_length, block_length):
                    _, mixed = forward_suffix(model, mixed, parts[target].block_ids[block_index:block_index + 1])
            else:
                mixed = splice_prefix_block(target_prefixes[target], relocated)
            repeated_prefix_tokens = 0
            repeated_prefix_start = None
            if repeat_prefix_kv:
                # target_prefixes[target] was encoded at positions 0..P-1.
                # It is now appended after target-prefix + transplanted-block,
                # so its post-RoPE Keys must move to P+B..P+B+P-1. Values are
                # position-independent and are copied by relocate_block.
                repeated_prefix_start = mixed[0][0].shape[-2]
                repeated_prefix = relocate_block(
                    model, target_prefixes[target], 0, repeated_prefix_start
                )
                repeated_prefix_tokens = parts[target].prefix_ids.numel()
                mixed = splice_prefix_block(mixed, repeated_prefix)
                del repeated_prefix
            logits, prompt_layers = forward_suffix(model, mixed, parts[target].suffix_ids)
            generation_layers = prompt_layers
            if placeholder_ids is not None:
                logits, generation_layers = forward_suffix(model, prompt_layers, placeholder_ids)
            tokens, hit_limit = greedy_continue(model, tokenizer, logits, generation_layers, generation_limit)
            sync_cuda(); elapsed = time.perf_counter() - started
            key_cosine, value_cosine = layer_token_cosines(relocated, full_blocks[target])
            matrices[condition] = {
                "key_before_rope": raw_key,
                "key_rope_corrected": key_cosine,
                "value": value_cosine,
            }
            result = result_from_generation(
                tokenizer,
                record["dataset"],
                record[f"gold_{target}"],
                tokens,
                hit_limit,
                elapsed + prefix_seconds[target],
                generation_limit,
                args.enable_thinking,
                args.explicit_reasoning,
                "The answer is: \\boxed{" if args.boxed_output and not args.explicit_reasoning else "",
            )
            result.update(logits_comparison(full_logits[target], logits))
            result.update({
                "source_side": "clean" if clean_reuse else source,
                "target_side": target,
                "source_block_start": source_start,
                "target_block_start": target_start,
                "rope_position_delta": target_start - source_start,
                "target_prefix_prefill_seconds": prefix_seconds[target],
                "reuse_suffix_and_generation_seconds": elapsed,
                "mean_key_cosine": float(key_cosine.mean()),
                "mean_key_cosine_before_rope": float(raw_key.mean()),
                "mean_value_cosine": float(value_cosine.mean()),
                "recomputed_tail_tokens": recomputed_tail_tokens,
                "repeated_target_prefix_tokens": repeated_prefix_tokens,
                "repeated_target_prefix_start": repeated_prefix_start,
            })
            reuse_results[condition] = result
            del relocated, mixed, prompt_layers, generation_layers, logits

        sample_dir = sample_artifact_root / safe_task_name(record["task_id"])
        if not run_reuse:
            npz_path = None
            plot_path = None
            matrix_shape = None
        elif args.no_plots:
            sample_dir.mkdir(parents=True, exist_ok=True)
            npz_path = sample_dir / "kv_similarity.npz"
            np.savez_compressed(
                npz_path,
                **{f"{condition}_{metric}": value for condition, values in matrices.items()
                   for metric, value in values.items()},
            )
            plot_path = None
        else:
            npz_path, plot_path = save_similarity_artifacts(sample_dir, matrices)
        if run_reuse:
            matrix_shape = list(next(iter(matrices.values()))["value"].shape)

        self_controls = {}
        for condition, side, _ in SELF_CONDITIONS:
            if condition not in reuse_results:
                continue
            reuse = reuse_results[condition]
            full = full_results[side]
            output_match = reuse["output_token_ids"] == full["output_token_ids"]
            passed = (
                output_match
                and reuse["first_token_kl_full_to_reuse"] <= args.self_kl_atol
                and reuse["first_token_logit_cosine"] >= args.self_logit_cos_min
            )
            self_controls[condition] = {
                "pass": passed,
                "prediction_match": reuse["prediction"] == full["prediction"],
                "output_token_ids_match": output_match,
                "first_token_kl": reuse["first_token_kl_full_to_reuse"],
                "first_token_logit_cosine": reuse["first_token_logit_cosine"],
                "max_abs_logit_diff": reuse["first_token_max_abs_logit_diff"],
            }
            if args.fail_on_self_mismatch and not passed:
                raise RuntimeError(f"{record['task_id']} {condition} failed the self control")

        row = {
            "sample_index": sample_index,
            "task_id": record["task_id"],
            "dataset": record["dataset"],
            "input": {
                "prefix_a": record["prefix_a"],
                "prefix_b": record["prefix_b"],
                "shared_block": record["shared_block"],
                "question": record["question"],
                "gold_a": record["gold_a"],
                "gold_b": record["gold_b"],
            },
            "tokenization": {
                side: {
                    "rendered_input": parts[side].rendered,
                    "full_input_token_ids": parts[side].full_ids.tolist(),
                    "prefix_tokens": parts[side].prefix_ids.numel(),
                    "block_tokens": parts[side].block_ids.numel(),
                    "suffix_tokens": parts[side].suffix_ids.numel(),
                    "block_token_span": [parts[side].block_token_start, parts[side].block_token_end],
                    "block_char_span": [parts[side].block_char_start, parts[side].block_char_end],
                    "cached_token_char_span": [parts[side].cached_char_start, parts[side].cached_char_end],
                }
                for side in ("a", "b")
            },
            "full": full_results,
            "reuse": reuse_results,
            "self_controls": self_controls,
            "artifacts": {
                "similarity_npz": str(npz_path) if npz_path is not None else None,
                "similarity_plot": str(plot_path) if plot_path is not None else None,
                "matrix_shape_tokens_by_layers": matrix_shape,
            },
        }
        append_jsonl(predictions_path, row)
        existing.append(row)
        completed.add(record["task_id"])
        write_json_atomic(summary_path, {"config": run_config, "metrics": summarize(existing)})
        status = f"[{sample_index}/{len(records)}] {record['dataset']} {record['task_id']}"
        if run_full:
            status += f" full={full_results['a']['prediction']}->{full_results['b']['prediction']}"
        if run_reuse:
            status += " reuse=" + ",".join(
                f"{name}:{reuse_results[name]['prediction']}" for name, _, _ in conditions
            )
        print(status, flush=True)
        del parts, full_logits, full_blocks, target_prefixes, matrices
        if clean_block is not None:
            del clean_block
        gc.collect(); torch.cuda.empty_cache()

    final_summary = {"config": run_config, "metrics": summarize(existing)}
    write_json_atomic(summary_path, final_summary)
    print(json.dumps(final_summary["metrics"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
