"""Bounded GPU prefix KV/state cache for a frozen eval-mode base model."""
from collections import OrderedDict
import torch
from scripts import batch_eval as batch_ops
from scripts import run_direct_reuse as base


class FrozenPrefixCache:
    def __init__(self, model, pooling='last', max_bytes=8 * 2**30):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('prefix cache requires a frozen eval model')
        self.model = model
        self.pooling = pooling
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    @torch.no_grad()
    def get_batch(self, prefixes):
        keys = [tuple(ids.tolist()) for ids in prefixes]
        missing = list(dict.fromkeys(k for k in keys if k not in self.entries))
        fresh = {}
        if missing:
            device = base.input_device(self.model)
            width = max(map(len, missing))
            ids = torch.zeros((len(missing), width),dtype=torch.long,device=device)
            attention = torch.zeros_like(ids); positions = torch.zeros_like(ids)
            for i, key in enumerate(missing):
                length = len(key)
                ids[i,-length:] = torch.tensor(key,dtype=torch.long,device=device)
                attention[i,-length:] = 1
                positions[i,-length:] = torch.arange(length,device=device)
            output = self.model.model(input_ids=ids,attention_mask=attention,
                position_ids=positions,cache_position=torch.arange(width,device=device),
                use_cache=True,return_dict=True)
            layers = base.extract_cache_tensors(output.past_key_values)
            for i,key in enumerate(missing):
                length = len(key)
                kv = [(k[i:i+1,...,-length:,:].clone().contiguous(),
                       v[i:i+1,...,-length:,:].clone().contiguous()) for k,v in layers]
                hidden = output.last_hidden_state[i,-length:]
                state = (hidden[-1] if self.pooling == 'last' else hidden.mean(dim=0)).detach().clone()
                size = state.numel()*state.element_size() + sum(
                    t.numel()*t.element_size() for pair in kv for t in pair)
                fresh[key] = (kv,state,size)
        # Keep this batch's references even if bounded insertion evicts a key.
        values = {k:self.entries[k] for k in keys if k in self.entries}
        self.hits += len(keys)-len(missing)
        self.misses += len(missing)
        values.update(fresh)
        for key,value in fresh.items():
            if value[2] > self.max_bytes:
                continue
            while self.entries and self.bytes+value[2] > self.max_bytes:
                _,old = self.entries.popitem(last=False)
                self.bytes -= old[2]
            self.entries[key] = value; self.bytes += value[2]
        for key in keys:
            if key in self.entries:
                self.entries.move_to_end(key)
        return [values[k][0] for k in keys], torch.stack([values[k][1] for k in keys])


@torch.no_grad()
def relocate_block_shared_rope(model, block, source_start, target_start):
    """Same arithmetic as baseline; compute positional cos/sin once per block."""
    if source_start == target_start:
        return [(key.clone(),value.clone().contiguous()) for key,value in block]
    ref = block[0][0]; length = ref.shape[-2]
    source = torch.arange(source_start,source_start+length,device=ref.device)
    target = torch.arange(target_start,target_start+length,device=ref.device)
    cos,sin = base.rope_cos_sin(model,ref,torch.cat((source,target)))
    cs,ct = cos[...,:length,:],cos[...,length:,:]
    ss,st = sin[...,:length,:],sin[...,length:,:]
    result = []
    for key,value in block:
        rotated = key.float()
        unrotated = rotated*cs-base.rotate_half_qwen(rotated)*ss
        relocated = unrotated*ct+base.rotate_half_qwen(unrotated)*st
        result.append((relocated.to(key.dtype).contiguous(),value.clone().contiguous()))
    return result