"""Differentiable universal soft suffix for cross-prefix KV reuse.

The base model is frozen.  A single trainable ``[soft_tokens, hidden_size]``
parameter is inserted after the transplanted shared-block cache and before the
ordinary target suffix.  Teacher logits come from a normal target-side Full
prefill; student logits come from the reused cache plus the soft suffix.
"""

from __future__ import annotations

from typing import Any
import time

import torch
import torch.nn.functional as F

from scripts import run_direct_reuse as base
from latent_remedy.tensor_batch import BatchedPast, project_spans


def continuation_logits_batch(
    model: Any,
    rows: list[tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]],
    soft_prompt: torch.Tensor | None,
    *, batched_past: BatchedPast | None = None, prepared_tokens=None,
) -> list[torch.Tensor]:
    """Teacher-forced continuation logits for heterogeneous reused prefixes.

    This is the differentiable counterpart of ``batch_eval.forward_suffix_batch``:
    it reuses that module's cache padding and DynamicCache installation rather
    than rebuilding the non-prefix-KV batching machinery in the trainer.
    """
    if not rows:
        return []
    from scripts import batch_eval as batch_ops

    device = base.input_device(model)
    embedding = model.get_input_embeddings()
    prefix_lengths = (batched_past.lengths if batched_past is not None
                      else [int(past[0][0].shape[-2]) for past, _, _ in rows])
    if len(prefix_lengths) != len(rows):
        raise ValueError("cache and continuation batch lengths differ")
    suffix_lengths = (prepared_tokens['suffix_lengths'] if prepared_tokens is not None
                      else [int(suffix.numel()) for _, suffix, _ in rows])
    answer_lengths = (prepared_tokens['answer_lengths'] if prepared_tokens is not None
                      else [int(answer.numel()) for _, _, answer in rows])
    if not all(suffix_lengths) or not all(answer_lengths):
        raise ValueError("batched continuation requires nonempty suffixes and answers")
    if soft_prompt is None:
        soft_length = 0
    elif soft_prompt.ndim == 2:
        soft_length = int(soft_prompt.shape[0])
        if soft_prompt.shape[1] != embedding.weight.shape[1]:
            raise ValueError("soft prompt must have shape [soft_tokens, hidden_size]")
    elif soft_prompt.ndim == 3:
        if soft_prompt.shape[0] != len(rows) or soft_prompt.shape[2] != embedding.weight.shape[1]:
            raise ValueError("batched soft prompt must have shape [batch, soft_tokens, hidden_size]")
        soft_length = int(soft_prompt.shape[1])
    else:
        raise ValueError("soft prompt must have two or three dimensions")
    input_lengths = [soft_length + suffix + answer - 1
                     for suffix, answer in zip(suffix_lengths, answer_lengths)]
    max_prefix, width = max(prefix_lengths), max(input_lengths)
    token_ids = prepared_tokens['ids'] if prepared_tokens is not None else torch.nn.utils.rnn.pad_sequence(
        [torch.cat((suffix.to(device), answer[:-1].to(device)))
         for _, suffix, answer in rows], batch_first=True, padding_value=0)
    inputs = embedding(token_ids)
    if soft_prompt is not None:
        prompt = soft_prompt.to(device=device, dtype=inputs.dtype)
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0).expand(len(rows), -1, -1)
        inputs = torch.cat((prompt, inputs), dim=1)
    lengths = torch.tensor(input_lengths, device=device)
    prefix = torch.tensor(prefix_lengths, device=device)
    columns = torch.arange(width, device=device)
    valid = columns[None, :] < lengths[:, None]
    positions = (prefix[:, None] + columns[None, :]).masked_fill(~valid, 0)
    prefix_mask = torch.arange(max_prefix, device=device)[None, :] >= (max_prefix - prefix[:, None])
    attention = torch.cat((prefix_mask, valid), dim=1).long()
    padded = batched_past.layers if batched_past is not None else batch_ops.pad_cache(
        [(past, length) for (past, _, _), length in zip(rows, prefix_lengths)], max_prefix
    )
    output = model.model(
        inputs_embeds=inputs,
        attention_mask=attention,
        position_ids=positions,
        cache_position=torch.arange(max_prefix, max_prefix + width, device=device),
        past_key_values=batch_ops.make_batch_cache(model, padded),
        use_cache=False,
        return_dict=True,
    )
    return project_spans(model, output.last_hidden_state, list(range(len(rows))),
                         [soft_length + n - 1 for n in suffix_lengths], answer_lengths,
                         packed=prepared_tokens is not None and prepared_tokens.get('packed',False))


def suffix_logits_batch(
    model: Any,
    rows: list[tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]],
    soft_prompt: torch.Tensor,
) -> list[torch.Tensor]:
    """Return logits predicting every suffix token after a per-row soft prompt.

    The final soft token predicts the first suffix token; each subsequent
    suffix token is teacher-forced to predict the next one.  This exposes the
    post-reuse prompt-reading trajectory for distillation, rather than only
    supervising generated answer tokens.
    """
    if not rows:
        return []
    if soft_prompt.ndim != 3 or soft_prompt.shape[0] != len(rows):
        raise ValueError("suffix distillation requires [batch, soft_tokens, hidden_size]")
    from scripts import batch_eval as batch_ops

    device = base.input_device(model)
    embedding = model.get_input_embeddings()
    if soft_prompt.shape[2] != embedding.weight.shape[1]:
        raise ValueError("soft prompt hidden size mismatch")
    soft_length = int(soft_prompt.shape[1])
    if soft_length < 1:
        raise ValueError("suffix distillation requires at least one soft token")
    prefix_lengths = [int(past[0][0].shape[-2]) for past, _ in rows]
    suffix_lengths = [int(suffix.numel()) for _, suffix in rows]
    if not all(suffix_lengths):
        raise ValueError("suffix distillation requires nonempty suffixes")
    max_prefix = max(prefix_lengths)
    width = soft_length + max(length - 1 for length in suffix_lengths)
    inputs = torch.zeros(
        (len(rows), width, embedding.weight.shape[1]), device=device, dtype=embedding.weight.dtype,
    )
    positions = torch.zeros((len(rows), width), device=device, dtype=torch.long)
    suffix_attention = torch.zeros((len(rows), width), device=device, dtype=torch.long)
    for index, (_, suffix_ids) in enumerate(rows):
        pieces = [soft_prompt[index].to(device=device, dtype=inputs.dtype)]
        if suffix_ids.numel() > 1:
            pieces.append(embedding(suffix_ids[:-1].unsqueeze(0).to(device))[0])
        sequence = torch.cat(pieces, dim=0)
        inputs[index, :sequence.shape[0]] = sequence
        positions[index, :sequence.shape[0]] = torch.arange(
            prefix_lengths[index], prefix_lengths[index] + sequence.shape[0], device=device,
        )
        suffix_attention[index, :sequence.shape[0]] = 1
    attention = torch.zeros((len(rows), max_prefix + width), device=device, dtype=torch.long)
    for index, length in enumerate(prefix_lengths):
        attention[index, max_prefix - length:max_prefix] = 1
        attention[index, max_prefix:] = suffix_attention[index]
    padded = batch_ops.pad_cache(
        [(past, length) for (past, _), length in zip(rows, prefix_lengths)], max_prefix,
    )
    output = model.model(
        inputs_embeds=inputs,
        attention_mask=attention,
        position_ids=positions,
        cache_position=torch.arange(max_prefix, max_prefix + width, device=device),
        past_key_values=batch_ops.make_batch_cache(model, padded),
        use_cache=False,
        return_dict=True,
    )
    results = []
    for index, suffix_length in enumerate(suffix_lengths):
        first = soft_length - 1
        results.append(model.lm_head(
            output.last_hidden_state[index, first:first + suffix_length]
        ).float())
    return results


def append_soft_suffix_batch(
    model: Any,
    mixed_rows: list[list[tuple[torch.Tensor, torch.Tensor]]],
    suffix_ids: list[torch.Tensor],
    soft_prompt: torch.Tensor,
) -> tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor]]], list[int]]:
    """Batch append one shared soft prompt and heterogeneous target suffixes.

    The returned per-row KV states are compatible with ``batch_eval.decode_batch``.
    """
    from scripts import batch_eval as batch_ops

    device = base.input_device(model)
    embedding = model.get_input_embeddings()
    prefix_lengths = [int(cache[0][0].shape[-2]) for cache in mixed_rows]
    suffix_lengths = [int(ids.numel()) for ids in suffix_ids]
    max_prefix = max(prefix_lengths)
    soft = soft_prompt.to(device=device, dtype=embedding.weight.dtype)
    soft_length = int(soft.shape[0])
    width = soft_length + max(suffix_lengths)
    inputs = torch.zeros((len(mixed_rows), width, embedding.weight.shape[1]),
                         device=device, dtype=embedding.weight.dtype)
    positions = torch.zeros((len(mixed_rows), width), device=device, dtype=torch.long)
    suffix_attention = torch.zeros((len(mixed_rows), width), device=device, dtype=torch.long)
    final_lengths = []
    for row, ids in enumerate(suffix_ids):
        suffix = embedding(ids.unsqueeze(0).to(device))[0]
        length = soft_length + suffix.shape[0]
        inputs[row, :soft_length] = soft
        inputs[row, soft_length:length] = suffix
        positions[row, :length] = torch.arange(prefix_lengths[row], prefix_lengths[row] + length,
                                                device=device)
        suffix_attention[row, :length] = 1
        final_lengths.append(prefix_lengths[row] + length)
    attention = torch.zeros((len(mixed_rows), max_prefix + width), device=device, dtype=torch.long)
    for row, prefix_length in enumerate(prefix_lengths):
        attention[row, max_prefix - prefix_length:max_prefix] = 1
        attention[row, max_prefix:] = suffix_attention[row]
    padded = batch_ops.pad_cache(list(zip(mixed_rows, prefix_lengths)), max_prefix)
    output = model(
        inputs_embeds=inputs,
        attention_mask=attention,
        position_ids=positions,
        cache_position=torch.arange(max_prefix, max_prefix + width, device=device),
        past_key_values=batch_ops.make_batch_cache(model, padded),
        use_cache=True,
        return_dict=True,
    )
    row_ids = torch.arange(len(mixed_rows), device=device)
    last_ids = torch.tensor([soft_length + length - 1 for length in suffix_lengths], device=device)
    logits = output.logits[row_ids, last_ids].detach().float()
    layers = base.extract_cache_tensors(output.past_key_values)
    trimmed = []
    for row, (prefix_length, suffix_length) in enumerate(zip(prefix_lengths, suffix_lengths)):
        start = max_prefix - prefix_length
        end = start + prefix_length + soft_length + suffix_length
        trimmed.append([
            (key[row:row + 1, ..., start:end, :].contiguous(),
             value[row:row + 1, ..., start:end, :].contiguous())
            for key, value in layers
        ])
    return logits, trimmed, final_lengths


def freeze_model(model: Any) -> None:
    """Freeze model weights while retaining gradients with respect to inputs."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def initialize_soft_prompt(
    model: Any,
    soft_tokens: int,
) -> torch.nn.Parameter:
    """Initialize every continuous suffix embedding exactly to zero."""
    if soft_tokens < 1:
        raise ValueError("soft_tokens must be positive")
    hidden_size = model.get_input_embeddings().weight.shape[1]
    return torch.nn.Parameter(
        torch.zeros((soft_tokens, hidden_size), device=base.input_device(model), dtype=torch.float32)
    )


def _prefill_no_grad(
    model: Any, ids: torch.Tensor
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Prefill using ``no_grad`` rather than inference tensors.

    The returned tensors are ordinary detached tensors and can safely be used
    as constants by the differentiable soft-suffix forward.
    """
    device = base.input_device(model)
    input_ids = ids.unsqueeze(0).to(device)
    positions = torch.arange(input_ids.shape[1], device=device, dtype=torch.long)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            use_cache=True,
            return_dict=True,
        )
        logits = output.logits[0, -1].detach().float().clone()
        layers = [
            (key.detach().clone(), value.detach().clone())
            for key, value in base.extract_cache_tensors(output.past_key_values, clone=False)
        ]
    return logits, layers


def build_reused_past(
    model: Any,
    target_parts: base.PromptParts,
    source_parts: base.PromptParts,
    timings_ms: dict[str, float] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Construct target-prefix + relocated donor-block KV as constants."""
    if not torch.equal(target_parts.block_ids, source_parts.block_ids):
        raise ValueError("shared block token IDs differ across source and target")
    def measure(name: str, operation: Any) -> Any:
        if timings_ms is None:
            return operation()
        torch.cuda.synchronize(base.input_device(model))
        started = time.perf_counter()
        result = operation()
        torch.cuda.synchronize(base.input_device(model))
        timings_ms[name] = (time.perf_counter() - started) * 1000
        return result

    source_logits_layers = measure(
        "source_prefill", lambda: _prefill_no_grad(model, source_parts.full_ids)
    )
    target_logits_layers = measure(
        "target_prefix_prefill", lambda: _prefill_no_grad(model, target_parts.prefix_ids)
    )
    _, source_layers = source_logits_layers
    _, target_layers = target_logits_layers
    source_block = base.slice_cache(
        source_layers, source_parts.block_token_start, source_parts.block_token_end
    )
    def relocate_and_splice() -> list[tuple[torch.Tensor, torch.Tensor]]:
        with torch.no_grad():
            relocated = base.relocate_block(
                model,
                source_block,
                source_parts.block_token_start,
                target_parts.block_token_start,
            )
            return base.splice_prefix_block(target_layers, relocated)

    mixed = measure("rope_relocate_and_splice", relocate_and_splice)
    # ``rope_cos_sin`` currently uses inference_mode.  Cloning here, outside
    # that context, materializes ordinary tensors safe for autograd saves.
    return [(key.clone(), value.clone()) for key, value in mixed]


def full_next_logits(model: Any, full_ids: torch.Tensor) -> torch.Tensor:
    """Return detached Full teacher logits for the first generated token."""
    logits, _ = _prefill_no_grad(model, full_ids)
    return logits


def continuation_logits(
    model: Any,
    past_layers: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_ids: torch.Tensor,
    answer_tokens: torch.Tensor,
    soft_prompt: torch.Tensor | None,
) -> torch.Tensor:
    """Teacher-forced logits for every token in ``answer_tokens``.

    The backbone processes soft vectors, the ordinary suffix, and all answer
    tokens except the final one.  Only answer-prediction hidden states are
    projected through the vocabulary head, avoiding a large
    ``continuation_length × vocab`` tensor for irrelevant prompt positions.
    """
    if suffix_ids.numel() == 0 or answer_tokens.numel() == 0:
        raise ValueError("suffix_ids and answer_tokens must be nonempty")
    device = base.input_device(model)
    embedding = model.get_input_embeddings()
    suffix_embeddings = embedding(suffix_ids.unsqueeze(0).to(device))
    answer_inputs = answer_tokens[:-1].unsqueeze(0).to(device)
    answer_embeddings = embedding(answer_inputs) if answer_inputs.numel() else None
    pieces = []
    soft_length = 0
    if soft_prompt is not None:
        if soft_prompt.ndim != 2 or soft_prompt.shape[1] != suffix_embeddings.shape[-1]:
            raise ValueError("soft prompt must have shape [soft_tokens, hidden_size]")
        soft = soft_prompt.to(device=device, dtype=suffix_embeddings.dtype).unsqueeze(0)
        pieces.append(soft)
        soft_length = soft.shape[1]
    pieces.append(suffix_embeddings)
    if answer_embeddings is not None:
        pieces.append(answer_embeddings)
    inputs_embeds = torch.cat(pieces, dim=1)
    past_length = int(past_layers[0][0].shape[-2])
    positions = torch.arange(
        past_length, past_length + inputs_embeds.shape[1], device=device, dtype=torch.long
    )
    output = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(
            (1, past_length + inputs_embeds.shape[1]), device=device, dtype=torch.long
        ),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=base.make_dynamic_cache(model, past_layers),
        use_cache=False,
        return_dict=True,
    )
    first_prediction = soft_length + int(suffix_ids.numel()) - 1
    prediction_hidden = output.last_hidden_state[
        0, first_prediction:first_prediction + int(answer_tokens.numel())
    ]
    if prediction_hidden.shape[0] != answer_tokens.numel():
        raise RuntimeError("failed to select all teacher-forced answer positions")
    return model.lm_head(prediction_hidden).float()


def append_suffix(
    model: Any,
    past_layers: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_ids: torch.Tensor,
    soft_prompt: torch.Tensor | None,
    *,
    return_cache: bool = False,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
    """Append optional soft vectors followed by the ordinary target suffix.

    This function deliberately has no ``no_grad`` decorator: during training,
    gradients must flow through all 64 soft positions to ``soft_prompt``.
    """
    if suffix_ids.numel() == 0:
        raise ValueError("target suffix must not be empty")
    device = base.input_device(model)
    token_embeddings = model.get_input_embeddings()(suffix_ids.unsqueeze(0).to(device))
    if soft_prompt is None:
        inputs_embeds = token_embeddings
        soft_length = 0
    else:
        if soft_prompt.ndim != 2 or soft_prompt.shape[1] != token_embeddings.shape[-1]:
            raise ValueError("soft prompt must have shape [soft_tokens, hidden_size]")
        soft = soft_prompt.to(device=device, dtype=token_embeddings.dtype).unsqueeze(0)
        inputs_embeds = torch.cat((soft, token_embeddings), dim=1)
        soft_length = soft.shape[1]
    past_length = int(past_layers[0][0].shape[-2])
    positions = torch.arange(
        past_length, past_length + inputs_embeds.shape[1], device=device, dtype=torch.long
    )
    output = model(
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(
            (1, past_length + inputs_embeds.shape[1]), device=device, dtype=torch.long
        ),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        past_key_values=base.make_dynamic_cache(model, past_layers),
        use_cache=return_cache,
        return_dict=True,
    )
    logits = output.logits[0, -1].float()
    layers = None
    if return_cache:
        layers = base.extract_cache_tensors(output.past_key_values)
        expected = past_length + soft_length + int(suffix_ids.numel())
        if layers[0][0].shape[-2] != expected:
            raise RuntimeError("soft suffix cache has an unexpected logical length")
    return logits, layers


def forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute forward KL ``KL(Full teacher || reuse student)`` in FP32."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    return F.kl_div(student_logp, teacher_logp.exp(), reduction="sum") * temperature**2


def selective_residual_kl(
    full_logits: torch.Tensor,
    direct_logits: torch.Tensor,
    soft_logits: torch.Tensor,
    *,
    temperature: float,
    offset_threshold: float,
    offset_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repair only high-Full-vs-Direct divergence while preserving Direct elsewhere.

    ``weight`` is detached: Full is used only as an offline training oracle to
    decide whether repair is needed, never as a differentiable route to model
    parameters.  For low-offset rows, the soft suffix is instead distilled to
    the Direct continuation, constraining it not to introduce a new change.
    """
    if offset_scale <= 0:
        raise ValueError("offset_scale must be positive")
    full_to_direct = forward_kl(full_logits, direct_logits, temperature=temperature)
    full_to_soft = forward_kl(full_logits, soft_logits, temperature=temperature)
    direct_to_soft = forward_kl(direct_logits, soft_logits, temperature=temperature)
    weight = torch.sigmoid((full_to_direct.detach() - offset_threshold) / offset_scale)
    return weight * full_to_soft + (1.0 - weight) * direct_to_soft, weight, full_to_direct, direct_to_soft
