"""Prefix-conditioned latent adapter for Repeat-prefix distillation."""
from __future__ import annotations
from typing import Any
import torch
from scripts import run_direct_reuse as base
from scripts import batch_eval as batch_ops


class PrefixLatentAdapter(torch.nn.Module):
    """Generate a sample-specific latent sequence from its target prefix."""
    def __init__(self, hidden_size: int, latent_tokens: int, bottleneck_size: int = 256) -> None:
        super().__init__()
        self.latent_tokens = latent_tokens
        self.hidden_size = hidden_size
        self.base = torch.nn.Parameter(torch.zeros(latent_tokens, hidden_size))
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, bottleneck_size),
            torch.nn.SiLU(),
            torch.nn.Linear(bottleneck_size, latent_tokens * hidden_size),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, prefix_state: torch.Tensor) -> torch.Tensor:
        prefix_state = prefix_state.to(dtype=self.base.dtype)
        return self.base + self.network(prefix_state).reshape(
            *prefix_state.shape[:-1], self.latent_tokens, self.hidden_size)


@torch.no_grad()
def target_prefix_state(model: Any, prefix_ids: torch.Tensor) -> torch.Tensor:
    device = base.input_device(model)
    input_ids = prefix_ids.unsqueeze(0).to(device)
    positions = torch.arange(input_ids.shape[1], device=device)
    output = model.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=positions.unsqueeze(0),
        cache_position=positions,
        use_cache=False,
        return_dict=True,
    )
    return output.last_hidden_state[0, -1].detach()


@torch.no_grad()
def target_prefix_states_batch(
    model: Any, prefixes: list[torch.Tensor], pooling: str = "last",
) -> torch.Tensor:
    """Return frozen prefix representations using last-token or mean pooling."""
    if pooling not in {"last", "mean"}:
        raise ValueError(f"unsupported prefix pooling: {pooling}")
    device = base.input_device(model)
    lengths = [int(prefix.numel()) for prefix in prefixes]
    width = max(lengths)
    input_ids = torch.zeros((len(prefixes), width), dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    positions = torch.zeros_like(input_ids)
    for row, prefix in enumerate(prefixes):
        length = lengths[row]
        input_ids[row, width - length:] = prefix.to(device)
        attention[row, width - length:] = 1
        positions[row, width - length:] = torch.arange(length, device=device)
    output = model.model(
        input_ids=input_ids, attention_mask=attention, position_ids=positions,
        cache_position=torch.arange(width, device=device), use_cache=False, return_dict=True,
    )
    if pooling == "last":
        return output.last_hidden_state[:, -1].detach()
    weights = attention.unsqueeze(-1).to(output.last_hidden_state.dtype)
    return ((output.last_hidden_state * weights).sum(dim=1) / weights.sum(dim=1)).detach()


@torch.no_grad()
def append_dynamic_latents_batch(
    model: Any, mixed_rows: list[list[tuple[torch.Tensor, torch.Tensor]]],
    suffix_ids: list[torch.Tensor], latents: torch.Tensor,
) -> tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor]]], list[int]]:
    """Append a distinct ``[K,d]`` latent sequence to every cache row."""
    device = base.input_device(model)
    embedding = model.get_input_embeddings()
    prefix_lengths = [int(cache[0][0].shape[-2]) for cache in mixed_rows]
    suffix_lengths = [int(ids.numel()) for ids in suffix_ids]
    batch, latent_length, hidden = latents.shape
    if batch != len(mixed_rows) or hidden != embedding.weight.shape[1]:
        raise ValueError("invalid dynamic latent batch shape")
    max_prefix, width = max(prefix_lengths), latent_length + max(suffix_lengths)
    inputs = torch.zeros((batch, width, hidden), device=device, dtype=embedding.weight.dtype)
    positions = torch.zeros((batch, width), device=device, dtype=torch.long)
    suffix_attention = torch.zeros((batch, width), device=device, dtype=torch.long)
    final_lengths = []
    for row, ids in enumerate(suffix_ids):
        suffix = embedding(ids.unsqueeze(0).to(device))[0]
        length = latent_length + suffix.shape[0]
        inputs[row, :latent_length] = latents[row].to(dtype=inputs.dtype)
        inputs[row, latent_length:length] = suffix
        positions[row, :length] = torch.arange(prefix_lengths[row], prefix_lengths[row] + length, device=device)
        suffix_attention[row, :length] = 1
        final_lengths.append(prefix_lengths[row] + length)
    attention = torch.zeros((batch, max_prefix + width), dtype=torch.long, device=device)
    for row, prefix_length in enumerate(prefix_lengths):
        attention[row, max_prefix - prefix_length:max_prefix] = 1
        attention[row, max_prefix:] = suffix_attention[row]
    padded = batch_ops.pad_cache(list(zip(mixed_rows, prefix_lengths)), max_prefix)
    output = model(
        inputs_embeds=inputs, attention_mask=attention, position_ids=positions,
        cache_position=torch.arange(max_prefix, max_prefix + width, device=device),
        past_key_values=batch_ops.make_batch_cache(model, padded), use_cache=True, return_dict=True,
    )
    logits = output.logits[
        torch.arange(batch, device=device),
        torch.tensor([latent_length + length - 1 for length in suffix_lengths], device=device),
    ].detach().float()
    layers = base.extract_cache_tensors(output.past_key_values)
    trimmed = []
    for row, (prefix_length, suffix_length) in enumerate(zip(prefix_lengths, suffix_lengths)):
        start, end = max_prefix - prefix_length, max_prefix + latent_length + suffix_length
        trimmed.append([(key[row:row + 1, ..., start:end, :].contiguous(), value[row:row + 1, ..., start:end, :].contiguous()) for key, value in layers])
    return logits, trimmed, final_lengths
