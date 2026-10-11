"""One batched dense forward supplies teacher logits, prefix states and donor KV."""
from collections import OrderedDict
import torch
from scripts import run_direct_reuse as base
from kv_semantic_attack.frozen_prefix_cache import relocate_block_shared_rope
from kv_semantic_attack.tensor_batch import BatchedPast, project_spans


class PairedFullContext:
    def __init__(self, model, pooling='last', max_bytes=16*2**30):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('requires frozen eval-mode model')
        self.model, self.pooling, self.max_bytes = model, pooling, max_bytes
        self.cache = OrderedDict()
        self.bytes = 0
        self.forward_calls = 0

    @torch.no_grad()
    def get_batch(self, targets, sources, traces, reasoning, *, tensorized=False, prepared_tokens=None):
        if not (len(targets)==len(sources)==len(traces)==len(reasoning)):
            raise ValueError('batch lengths differ')
        if tensorized:
            if self.max_bytes:
                raise ValueError('tensorized paired flow is batch-local, not a dataset cache')
            return self._get_tensor_batch(targets, sources, traces, reasoning, prepared_tokens)
        # Prefix+block identifies fixed context; suffix/prompt mode only affects
        # teacher input, not preceding KV under causal attention.
        key = lambda p: tuple(p.full_ids[:p.block_token_end].tolist())
        tk, sk = [key(p) for p in targets], [key(p) for p in sources]
        needed = {}
        teacher_rows = []
        for i,(p,trace,mode) in enumerate(zip(targets,traces,reasoning)):
            if mode:
                teacher_rows.append((i,p,torch.cat((p.full_ids,trace[:-1].to(p.full_ids.device))),len(trace)))
                needed[tk[i]] = p
        selected_keys = set(needed)
        for p,k in zip(targets+sources,tk+sk):
            if k not in self.cache and k not in selected_keys:
                needed[k] = p; selected_keys.add(k)
        requests = [(key(p),p,seq,i,n) for i,p,seq,n in teacher_rows]
        teacher_keys = {k for k,_,_,_,_ in requests}
        requests.extend((k,p,p.full_ids,None,0) for k,p in needed.items() if k not in teacher_keys)
        local = {k:self.cache[k] for k in tk+sk if k in self.cache}
        teacher = [None]*len(targets)
        if requests:
            device = base.input_device(self.model)
            width = max(len(seq) for _,_,seq,_,_ in requests)
            ids = torch.zeros((len(requests),width),dtype=torch.long,device=device)
            mask = torch.zeros_like(ids); positions = torch.zeros_like(ids)
            for row,(_,p,seq,i,n) in enumerate(requests):
                ids[row,-len(seq):] = seq.to(device)
                mask[row,-len(seq):] = 1
                positions[row,-len(seq):] = torch.arange(len(seq),device=device)
            output = self.model.model(input_ids=ids,attention_mask=mask,position_ids=positions,
                cache_position=torch.arange(width,device=device),use_cache=True,return_dict=True)
            self.forward_calls += 1
            layers = base.extract_cache_tensors(output.past_key_values)
            spans = [(row, width-len(seq)+len(p.full_ids)-1, n, i)
                     for row,(_,p,seq,i,n) in enumerate(requests) if i is not None]
            projected = project_spans(self.model, output.last_hidden_state,
                                     [x[0] for x in spans], [x[1] for x in spans],
                                     [x[2] for x in spans])
            for span, prediction in zip(spans, projected):
                teacher[span[3]] = prediction
            for row,(k,p,seq,i,n) in enumerate(requests):
                offset = width-len(seq)
                if k in local:
                    continue
                end = offset+p.block_token_end
                kv = [(a[row:row+1,...,offset:end,:].clone().contiguous(),
                       b[row:row+1,...,offset:end,:].clone().contiguous()) for a,b in layers]
                length = len(p.prefix_ids)
                prefix_hidden = output.last_hidden_state[row,offset:offset+length]
                state = (prefix_hidden[-1] if self.pooling=='last' else prefix_hidden.mean(0)).detach().clone()
                size = sum(v.numel()*v.element_size() for pair in kv for v in pair)+state.numel()*state.element_size()
                local[k] = (kv,state,size)
                if size <= self.max_bytes:
                    while self.cache and self.bytes+size>self.max_bytes:
                        _,old = self.cache.popitem(last=False); self.bytes -= old[2]
                    self.cache[k] = local[k]; self.bytes += size
        pasts, states = [], []
        for p,s,k,donor_key in zip(targets,sources,tk,sk):
            if not torch.equal(p.block_ids,s.block_ids):
                raise ValueError('shared block token mismatch')
            prefix = base.slice_cache(local[k][0],0,len(p.prefix_ids))
            block = base.slice_cache(local[donor_key][0],s.block_token_start,s.block_token_end)
            block = relocate_block_shared_rope(self.model,block,s.block_token_start,p.block_token_start)
            pasts.append(base.splice_prefix_block(prefix,block))
            states.append(local[k][1])
        return pasts, torch.stack(states), teacher

    @torch.no_grad()
    def _get_tensor_batch(self, targets, sources, traces, reasoning, prepared_tokens=None):
        """One dense forward, then batched role gather/RoPE per decoder layer.

        Keep only a layer loop: no sample-by-layer clone, relocation or cat.
        The result is already left padded and passed directly to the student.
        """
        if not targets:
            raise ValueError('empty paired batch')
        device = base.input_device(self.model)
        if prepared_tokens is not None:
            ids = prepared_tokens['ids']
            sequence_lengths = prepared_tokens['sequence_lengths']
            spans = prepared_tokens['spans']
            target_indices = prepared_tokens['target_rows']
            donor_indices = prepared_tokens['donor_rows']
        else:
            key = lambda p: tuple(p.full_ids[:p.block_token_end].tolist())
            requests, lookup, spans = [], {}, []
            for i, (p, trace, mode) in enumerate(zip(targets, traces, reasoning)):
                if mode:
                    seq = torch.cat((p.full_ids, trace[:-1].to(p.full_ids.device)))
                    lookup[key(p)] = len(requests)
                    spans.append((len(requests), len(p.full_ids)-1, len(trace), i))
                    requests.append((p, seq))
            for p in targets + sources:
                if key(p) not in lookup:
                    lookup[key(p)] = len(requests)
                    requests.append((p, p.full_ids))
            for p, s in zip(targets, sources):
                if not torch.equal(p.block_ids, s.block_ids):
                    raise ValueError('shared block token mismatch')
            sequence_lengths = [len(seq) for _, seq in requests]
            ids = torch.nn.utils.rnn.pad_sequence(
                [seq.to(device).flip(0) for _, seq in requests], batch_first=True).flip(1)
            target_indices = [lookup[key(p)] for p in targets]
            donor_indices = [lookup[key(p)] for p in sources]
        width = max(sequence_lengths)
        lengths = torch.tensor(sequence_lengths, device=device)
        offsets = width - lengths
        cols = torch.arange(width, device=device)
        mask = cols[None, :] >= offsets[:, None]
        positions = (cols[None, :] - offsets[:, None]).clamp_min(0)
        output = self.model.model(input_ids=ids, attention_mask=mask.long(),
            position_ids=positions, cache_position=cols, use_cache=True, return_dict=True)
        self.forward_calls += 1
        teacher = [None] * len(targets)
        projected = project_spans(self.model, output.last_hidden_state,
            [x[0] for x in spans], [width-sequence_lengths[x[0]]+x[1] for x in spans],
            [x[2] for x in spans],packed=prepared_tokens is not None and prepared_tokens.get('packed',False))
        if prepared_tokens is not None and prepared_tokens.get('packed',False):
            projected.example_ids = [x[3] for x in spans]
            teacher = projected
        else:
            for span, prediction in zip(spans, projected):
                teacher[span[3]] = prediction
        target_rows = torch.tensor(target_indices, device=device)
        donor_rows = torch.tensor(donor_indices, device=device)
        prefix_sizes = (prepared_tokens['prefix_lengths'] if prepared_tokens is not None
                else [len(p.prefix_ids) for p in targets])
        prefix_lengths = torch.tensor(prefix_sizes, device=device)
        true_lengths = (prepared_tokens['true_lengths'] if prepared_tokens is not None
                else [p.block_token_end for p in targets])
        cache_width = max(true_lengths)
        total = torch.tensor(true_lengths, device=device)
        col = torch.arange(cache_width, device=device)[None, :]
        logical = col - (cache_width-total[:, None])
        valid = logical >= 0
        is_block = logical >= prefix_lengths[:, None]
        source_starts = (prepared_tokens['source_starts'] if prepared_tokens is not None
                 else [p.block_token_start for p in sources])
        target_starts = (prepared_tokens['target_starts'] if prepared_tokens is not None
                 else [p.block_token_start for p in targets])
        source_start = torch.tensor(source_starts, device=device)
        target_start = torch.tensor(target_starts, device=device)
        delta = source_start - target_start
        row_indices = torch.where(is_block, donor_rows[:, None], target_rows[:, None])
        source_pos = logical + torch.where(is_block, delta[:, None], 0)
        gather_positions = (source_pos + offsets[row_indices]).clamp_min(0)
        layers = base.extract_cache_tensors(output.past_key_values)
        shift = is_block & valid & (delta[:, None] != 0)
        relocation = None
        if source_starts != target_starts:
            # One rotary call for all samples and all decoder layers.
            all_positions = torch.cat((source_pos.clamp_min(0).flatten(), logical.clamp_min(0).flatten()))
            cos, sin = base.rope_cos_sin(self.model, layers[0][0], all_positions)
            count = len(targets)*cache_width
            cs, ct = cos.reshape(-1, cos.shape[-1]).split(count)
            ss, st = sin.reshape(-1, sin.shape[-1]).split(count)
            relocation = [x.reshape(len(targets),cache_width,-1).unsqueeze(1)
                          for x in (cs,ct,ss,st)]
        padded = []
        for keys, values in layers:
            k = keys[row_indices, :, gather_positions, :].transpose(1,2)
            v = values[row_indices, :, gather_positions, :].transpose(1,2)
            if relocation is not None:
                cs, ct, ss, st = relocation
                unrotated = k.float()*cs-base.rotate_half_qwen(k.float())*ss
                relocated = (unrotated*ct+base.rotate_half_qwen(unrotated)*st).to(k.dtype)
                k = torch.where(shift[:,None,:,None], relocated, k)
            padded.append((k.masked_fill(~valid[:,None,:,None],0).contiguous(),
                           v.masked_fill(~valid[:,None,:,None],0).contiguous()))
        hidden = output.last_hidden_state
        if self.pooling == 'last':
            states = hidden[target_rows, offsets[target_rows]+prefix_lengths-1]
        else:
            prefix_cols = torch.arange(max(prefix_sizes), device=device)
            prefix_mask = prefix_cols[None,:] < prefix_lengths[:,None]
            h = hidden[target_rows[:,None],
                       (offsets[target_rows,None]+prefix_cols[None,:]).clamp_max(width-1)]
            states = (h*prefix_mask[:,:,None]).sum(1)/prefix_lengths[:,None]
        return BatchedPast(padded,true_lengths), states.detach(), teacher