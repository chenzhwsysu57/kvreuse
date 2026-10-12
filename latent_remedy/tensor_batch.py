"""Packed projections and left-padded KV, without sample-wise GPU kernels."""
from dataclasses import dataclass
import torch


@dataclass
class BatchedPast:
    layers: list[tuple[torch.Tensor, torch.Tensor]]
    lengths: list[int]


@dataclass
class PackedLogits:
    logits: torch.Tensor
    lengths: list[int]
    example_ids: list[int]


def project_spans(model, hidden, rows, starts, lengths, *, packed=False):
    """Project only valid answer positions in one lm_head call.

    Python metadata describes ragged spans; the gather and vocabulary projection
    operate on all valid tokens together, without padded vocabulary tensors.
    """
    if not lengths:
        return []
    device = hidden.device
    counts = torch.tensor(lengths, device=device, dtype=torch.long)
    span = torch.repeat_interleave(torch.arange(len(lengths), device=device), counts)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))
    within = torch.arange(sum(lengths), device=device) - offsets[span]
    row = torch.tensor(rows, device=device, dtype=torch.long)[span]
    position = torch.tensor(starts, device=device, dtype=torch.long)[span] + within
    logits = model.lm_head(hidden[row, position]).float()
    if packed:
        return PackedLogits(logits,list(lengths),list(range(len(lengths))))
    return list(logits.split(lengths))