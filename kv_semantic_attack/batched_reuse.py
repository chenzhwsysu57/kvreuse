"""Memory-conscious vectorized direct KV reuse for homogeneous batches."""

from __future__ import annotations

from typing import Any
import time

import torch


def _raw_prefill(batch: Any, model: Any, ids_list: list[torch.Tensor]):
    """Prefill once and retain only the batched cache, never per-row clones."""
    device = batch.base.input_device(model)
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
    output = model(input_ids=input_ids, attention_mask=attention, position_ids=positions,
                   cache_position=torch.arange(width, device=device), use_cache=True, return_dict=True)
    # Match the reference runner: detach cache tensors from the model output
    # before freeing it. This is one batch-level copy, not a list of row-level
    # copies, and avoids cache storage aliasing across forward calls.
    layers = batch.base.extract_cache_tensors(output.past_key_values)
    del output, input_ids, attention, positions
    return layers, lengths, width


def _gather_span(tensor: torch.Tensor, sequence_lengths: torch.Tensor, starts: torch.Tensor,
                 lengths: torch.Tensor, width: int):
    """Gather per-row spans from left-padded cache storage into a dense tensor."""
    maximum = int(lengths.max().item())
    relative = torch.arange(maximum, device=tensor.device)
    physical = width - sequence_lengths[:, None] + starts[:, None] + relative
    valid = relative[None, :] < lengths[:, None]
    physical = physical.clamp(0, tensor.shape[-2] - 1)
    index = physical[:, None, :, None].expand(-1, tensor.shape[1], -1, tensor.shape[-1])
    gathered = torch.gather(tensor, -2, index)
    return gathered * valid[:, None, :, None], valid


def _relocate_keys(batch: Any, model: Any, keys: torch.Tensor, valid: torch.Tensor,
                   source_starts: torch.Tensor, target_starts: torch.Tensor):
    """Apply source→target RoPE relocation for every row and layer together."""
    length = keys.shape[-2]
    positions = torch.arange(length, device=keys.device)
    source_positions = source_starts[:, None] + positions
    target_positions = target_starts[:, None] + positions
    rotary = model.model.rotary_emb
    dummy = torch.zeros((keys.shape[0], length, keys.shape[-1]), device=keys.device, dtype=keys.dtype)
    cos, sin = rotary(dummy, torch.cat((source_positions, target_positions), dim=1))
    cos, sin = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()
    cos_source, cos_target = cos[..., :length, :], cos[..., length:, :]
    sin_source, sin_target = sin[..., :length, :], sin[..., length:, :]
    source = keys.float()
    unrotated = source * cos_source - batch.base.rotate_half_qwen(source) * sin_source
    relocated = unrotated * cos_target + batch.base.rotate_half_qwen(unrotated) * sin_target
    return relocated.to(dtype=keys.dtype) * valid[:, None, :, None]


def _assemble_mixed(batch: Any, model: Any, source_layers, source_lengths, source_width,
                    target_layers, target_lengths, target_width, source_parts, target_parts):
    device = source_lengths.device
    block_starts = torch.tensor([part.block_token_start for part in source_parts], device=device)
    block_ends = torch.tensor([part.block_token_end for part in source_parts], device=device)
    block_lengths = block_ends - block_starts
    prefix_lengths = torch.tensor([int(part.prefix_ids.numel()) for part in target_parts], device=device)
    maximum_prefix = int(prefix_lengths.max().item())
    maximum_block = int(block_lengths.max().item())
    logical_lengths = prefix_lengths + block_lengths
    maximum_length = int(logical_lengths.max().item())
    # Allocate each layer only once; its valid sequence is left-padded to the
    # maximum logical length expected by the suffix/decode batch.
    mixed = []
    for (source_key, source_value), (target_key, target_value) in zip(source_layers, target_layers):
        prefix_key, prefix_valid = _gather_span(target_key, target_lengths, torch.zeros_like(prefix_lengths), prefix_lengths, target_width)
        prefix_value, _ = _gather_span(target_value, target_lengths, torch.zeros_like(prefix_lengths), prefix_lengths, target_width)
        block_key, block_valid = _gather_span(source_key, source_lengths, block_starts, block_lengths, source_width)
        block_value, _ = _gather_span(source_value, source_lengths, block_starts, block_lengths, source_width)
        block_key = _relocate_keys(batch, model, block_key, block_valid, block_starts, prefix_lengths)
        key = torch.zeros((source_key.shape[0], source_key.shape[1], maximum_length, source_key.shape[-1]),
                          dtype=source_key.dtype, device=device)
        value = torch.zeros_like(key)
        for row in range(key.shape[0]):
            start = maximum_length - int(logical_lengths[row])
            prefix_length = int(prefix_lengths[row])
            block_length = int(block_lengths[row])
            key[row, :, start:start + prefix_length] = prefix_key[row, :, :prefix_length]
            value[row, :, start:start + prefix_length] = prefix_value[row, :, :prefix_length]
            key[row, :, start + prefix_length:start + prefix_length + block_length] = block_key[row, :, :block_length]
            value[row, :, start + prefix_length:start + prefix_length + block_length] = block_value[row, :, :block_length]
        mixed.append((key, value))
    return mixed, logical_lengths


def _scatter_indices(lengths: torch.Tensor, maximum: int, *, offset: int = 0):
    """Destination positions plus a sentinel for invalid padded source slots."""
    positions = torch.arange(maximum, device=lengths.device)
    valid = positions[None, :] < lengths[:, None]
    destination = maximum - lengths[:, None] + offset + positions[None, :]
    # Scatter invalid values into an extra sentinel column, then discard it.
    return torch.where(valid, destination, torch.full_like(destination, maximum + offset)), valid


def _assemble_mixed_scatter(batch: Any, model: Any, source_layers, source_lengths, source_width,
                            target_layers, target_lengths, target_width, source_parts, target_parts):
    """Assemble each layer through two batch-wide scatter operations.

    This removes the per-row Python writes used by ``_assemble_mixed``. The
    extra destination slot makes invalid padded positions harmless even when a
    row's valid cache occupies index zero.
    """
    device = source_lengths.device
    block_starts = torch.tensor([part.block_token_start for part in source_parts], device=device)
    block_ends = torch.tensor([part.block_token_end for part in source_parts], device=device)
    block_lengths = block_ends - block_starts
    prefix_lengths = torch.tensor([int(part.prefix_ids.numel()) for part in target_parts], device=device)
    logical_lengths = prefix_lengths + block_lengths
    maximum_length = int(logical_lengths.max().item())
    max_prefix = int(prefix_lengths.max().item())
    max_block = int(block_lengths.max().item())
    prefix_destination, _ = _scatter_indices(prefix_lengths, max_prefix, offset=maximum_length - max_prefix)
    # Prefix begins at each row's left-padded logical start, not at a shared offset.
    prefix_destination = maximum_length - logical_lengths[:, None] + torch.arange(max_prefix, device=device)[None, :]
    prefix_valid = torch.arange(max_prefix, device=device)[None, :] < prefix_lengths[:, None]
    prefix_destination = torch.where(prefix_valid, prefix_destination,
                                     torch.full_like(prefix_destination, maximum_length))
    block_destination = maximum_length - logical_lengths[:, None] + prefix_lengths[:, None] + torch.arange(max_block, device=device)[None, :]
    block_valid_destination = torch.arange(max_block, device=device)[None, :] < block_lengths[:, None]
    block_destination = torch.where(block_valid_destination, block_destination,
                                    torch.full_like(block_destination, maximum_length))
    mixed = []
    for (source_key, source_value), (target_key, target_value) in zip(source_layers, target_layers):
        prefix_key, _ = _gather_span(target_key, target_lengths, torch.zeros_like(prefix_lengths), prefix_lengths, target_width)
        prefix_value, _ = _gather_span(target_value, target_lengths, torch.zeros_like(prefix_lengths), prefix_lengths, target_width)
        block_key, block_valid = _gather_span(source_key, source_lengths, block_starts, block_lengths, source_width)
        block_value, _ = _gather_span(source_value, source_lengths, block_starts, block_lengths, source_width)
        block_key = _relocate_keys(batch, model, block_key, block_valid, block_starts, prefix_lengths)
        shape = (source_key.shape[0], source_key.shape[1], maximum_length + 1, source_key.shape[-1])
        key = torch.zeros(shape, dtype=source_key.dtype, device=device)
        value = torch.zeros_like(key)
        prefix_index = prefix_destination[:, None, :, None].expand(-1, key.shape[1], -1, key.shape[-1])
        block_index = block_destination[:, None, :, None].expand(-1, key.shape[1], -1, key.shape[-1])
        key.scatter_(2, prefix_index, prefix_key).scatter_(2, block_index, block_key)
        value.scatter_(2, prefix_index, prefix_value).scatter_(2, block_index, block_value)
        mixed.append((key[..., :maximum_length, :], value[..., :maximum_length, :]))
    return mixed, logical_lengths


def _forward_suffix(batch: Any, model: Any, mixed_layers, logical_lengths, suffix_ids):
    device = batch.base.input_device(model)
    suffix_lengths = torch.tensor([int(ids.numel()) for ids in suffix_ids], device=device)
    width = int(suffix_lengths.max().item())
    maximum_past = mixed_layers[0][0].shape[-2]
    input_ids = torch.zeros((len(suffix_ids), width), dtype=torch.long, device=device)
    suffix_mask = torch.zeros_like(input_ids)
    positions = torch.zeros_like(input_ids)
    for row, ids in enumerate(suffix_ids):
        length = int(ids.numel())
        input_ids[row, :length] = ids.to(device)
        suffix_mask[row, :length] = 1
        positions[row, :length] = torch.arange(int(logical_lengths[row]), int(logical_lengths[row]) + length, device=device)
    attention = torch.zeros((len(suffix_ids), maximum_past + width), dtype=torch.long, device=device)
    for row, length in enumerate(logical_lengths.tolist()):
        attention[row, maximum_past - length:maximum_past] = 1
        attention[row, maximum_past:] = suffix_mask[row]
    output = model(input_ids=input_ids, attention_mask=attention, position_ids=positions,
                   cache_position=torch.arange(maximum_past, maximum_past + width, device=device),
                   past_key_values=batch.make_batch_cache(model, mixed_layers), use_cache=True, return_dict=True)
    last = suffix_lengths - 1
    logits = output.logits[torch.arange(len(suffix_ids), device=device), last].detach().float()
    layers = batch.base.extract_cache_tensors(output.past_key_values, clone=False)
    return logits, layers, logical_lengths + suffix_lengths


def reuse_rows_vectorized(batch: Any, model: Any, tokenizer: Any, rows, max_new_tokens: int,
                          assembly: str = "loop", phase_timing: dict[str, float] | None = None):
    """Direct reuse with batched relocation/splicing and no per-row KV clones."""
    target_parts, source_parts, records, sides = [], [], [], []
    for record, side in rows:
        source = "a" if side == "b" else "b"
        parts = {name: batch.build_parts(tokenizer, record, name, "reuse") for name in ("a", "b")}
        if not torch.equal(parts["a"].block_ids, parts["b"].block_ids):
            raise ValueError(f"{record['task_id']}: block tokenization mismatch")
        target_parts.append(parts[side])
        source_parts.append(parts[source])
        records.append(record)
        sides.append(side)
    def measure(name, callback):
        if phase_timing is None:
            return callback()
        torch.cuda.synchronize()
        started = time.perf_counter()
        value = callback()
        torch.cuda.synchronize()
        phase_timing[name] = phase_timing.get(name, 0.0) + time.perf_counter() - started
        return value

    source_layers, source_lengths, source_width = measure(
        "donor_prefill", lambda: _raw_prefill(batch, model, [part.full_ids for part in source_parts])
    )
    target_layers, target_lengths, target_width = measure(
        "target_prefix_prefill", lambda: _raw_prefill(batch, model, [part.prefix_ids for part in target_parts])
    )
    assembler = _assemble_mixed if assembly == "loop" else _assemble_mixed_scatter if assembly == "scatter" else None
    if assembler is None:
        raise ValueError(f"unsupported assembly: {assembly}")
    mixed_layers, logical_lengths = measure(
        "relocate_and_splice", lambda: assembler(
            batch, model, source_layers, source_lengths, source_width, target_layers, target_lengths,
            target_width, source_parts, target_parts,
        )
    )
    del source_layers, target_layers
    logits, layers, lengths = measure(
        "suffix_prefill", lambda: _forward_suffix(
            batch, model, mixed_layers, logical_lengths, [part.suffix_ids for part in target_parts]
        )
    )
    del mixed_layers
    # The result cache is already a single left-padded batch; no second
    # pad_cache()/cat() allocation is needed before decoding.
    generated = measure("decode", lambda: batch.decode_batch(
        model, tokenizer, logits, layers, lengths.tolist(), max_new_tokens
    ))
    output = []
    for record, side, tokens in zip(records, sides, generated):
        output.append(batch.base.result_from_generation(
            tokenizer, record["dataset"], record[f"gold_{side}"], tokens,
            len(tokens) >= max_new_tokens, 0.0, max_new_tokens, False, False,
            "The answer is: \\boxed{",
        ))
    return output