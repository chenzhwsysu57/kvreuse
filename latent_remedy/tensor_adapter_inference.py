"""Batch-wide Adapter prefill and greedy decode; only decoder-layer/token loops."""
import torch
from scripts import run_direct_reuse as base
from scripts import batch_eval as batch_ops
from latent_remedy.paired_full_context import PairedFullContext
from latent_remedy.batched_reuse import _left_pad_suffix_cache


@torch.no_grad()
def context_batch(model, targets, sources, pooling='last'):
    """One decoder call supplies prefix state/KV and donor block KV.

    Prefill only through the shared block, never target suffix or teacher answers.
    Integer context IDs and shared-block checks are CPU metadata preparation.
    """
    keys=lambda p:tuple(p.full_ids[:p.block_token_end].tolist())
    lookup={};sequences=[]
    for p in targets+sources:
        key=keys(p)
        if key not in lookup:
            lookup[key]=len(sequences)
            sequences.append(p.full_ids[:p.block_token_end].cpu())
    for p,s in zip(targets,sources):
        if not torch.equal(p.block_ids,s.block_ids):
            raise ValueError('shared block token mismatch')
    lengths=[len(seq) for seq in sequences]
    ids=torch.nn.utils.rnn.pad_sequence([seq.flip(0) for seq in sequences],batch_first=True).flip(1)
    prepared={'ids':ids.to(base.input_device(model)), 'sequence_lengths':lengths,'spans':[],
        'target_rows':[lookup[keys(p)] for p in targets],
        'donor_rows':[lookup[keys(s)] for s in sources],
        'prefix_lengths':[len(p.prefix_ids) for p in targets],
        'true_lengths':[p.block_token_end for p in targets],
        'source_starts':[s.block_token_start for s in sources],
        'target_starts':[p.block_token_start for p in targets],'packed':False}
    past,states,_=PairedFullContext(model,pooling,max_bytes=0).get_batch(
        targets,sources,[None]*len(targets),[False]*len(targets),tensorized=True,prepared_tokens=prepared)
    return past,states


@torch.no_grad()
def append_latents(model, past, suffixes, latents):
    """One embedding/decoder/head call, then batch-wide KV compaction."""
    device=base.input_device(model)
    batch=len(suffixes)
    if not batch or latents.ndim!=3 or latents.shape[0]!=batch:
        raise ValueError('invalid latent/suffix batch')
    suffix_lengths=[len(ids) for ids in suffixes]
    if not all(suffix_lengths):raise ValueError('suffixes must be nonempty')
    tokens=torch.nn.utils.rnn.pad_sequence([ids.cpu() for ids in suffixes],batch_first=True).to(device)
    embedding=model.get_input_embeddings()
    soft_length=latents.shape[1]
    inputs=torch.cat((latents.to(embedding.weight.dtype),embedding(tokens)),dim=1)
    lengths=torch.tensor(suffix_lengths,device=device)+soft_length
    prefix=torch.tensor(past.lengths,device=device)
    maximum=past.layers[0][0].shape[-2]
    width=inputs.shape[1]
    col=torch.arange(width,device=device)
    valid=col[None,:]<lengths[:,None]
    positions=(prefix[:,None]+col[None,:]).masked_fill(~valid,0)
    prefix_mask=torch.arange(maximum,device=device)[None,:]>=maximum-prefix[:,None]
    output=model.model(inputs_embeds=inputs,attention_mask=torch.cat((prefix_mask,valid),dim=1).long(),
        position_ids=positions,cache_position=torch.arange(maximum,maximum+width,device=device),
        past_key_values=batch_ops.make_batch_cache(model,past.layers),use_cache=True,return_dict=True)
    hidden=output.last_hidden_state[torch.arange(batch,device=device),lengths-1]
    logits=model.lm_head(hidden).float()
    layers=base.extract_cache_tensors(output.past_key_values,clone=False)
    compact=_left_pad_suffix_cache(layers,prefix,lengths,maximum)
    return logits,compact,(prefix+lengths).tolist()


@torch.no_grad()
def decode_tensor_batch(model, tokenizer, logits, layers, lengths, max_new_tokens,
                        stop_at_closing_brace=None):
    """Greedy token loop; all stop checks/positions/output writes are tensors.

    Finished rows remain in the rectangular decoder batch but their output is
    frozen at the first stop. No per-row GPU scalar reads until final transfer.
    """
    if max_new_tokens<1:raise ValueError('max_new_tokens must be positive')
    device=logits.device
    batch=len(lengths);maximum=layers[0][0].shape[-2]
    true_lengths=torch.tensor(lengths,device=device)
    stop=torch.tensor(stop_at_closing_brace or [False]*batch,device=device,dtype=torch.bool)
    if stop.numel()!=batch:raise ValueError('stop flags batch mismatch')
    eos=tokenizer.eos_token_id
    eos_ids=[eos] if isinstance(eos,int) else list(eos or [])
    brace_ids=tokenizer.encode('}',add_special_tokens=False)
    if not brace_ids:raise ValueError('tokenizer has no closing brace tokenization')
    eos_tensor=torch.tensor(eos_ids,device=device,dtype=torch.long)
    brace_tensor=torch.tensor(brace_ids,device=device,dtype=torch.long)
    def finished(token):
        return torch.isin(token,eos_tensor)|(stop & torch.isin(token,brace_tensor))
    generated=torch.zeros((batch,max_new_tokens),device=device,dtype=torch.long)
    current=logits.argmax(-1)
    generated[:,0]=current
    done=finished(current)
    output_lengths=torch.ones(batch,device=device,dtype=torch.long)
    attention=torch.ones((batch,maximum+max_new_tokens),device=device,dtype=torch.long)
    attention[:,:maximum]=(torch.arange(maximum,device=device)[None,:]>=maximum-true_lengths[:,None]).long()
    cache=batch_ops.make_batch_cache(model,layers)
    filler=eos_ids[0] if eos_ids else 0
    for step in range(1,max_new_tokens):
        if bool(done.all()):break
        output=model.model(input_ids=current[:,None],attention_mask=attention[:,:maximum+step],
            position_ids=(true_lengths+step-1)[:,None],
            cache_position=torch.tensor([maximum+step-1],device=device),
            past_key_values=cache,use_cache=True,return_dict=True)
        cache=output.past_key_values
        nxt=model.lm_head(output.last_hidden_state[:,-1]).argmax(-1)
        active=~done
        generated[:,step]=torch.where(active,nxt,filler)
        output_lengths+=active.long()
        done|=finished(nxt)
        current=torch.where(done,filler,nxt)
    tokens=generated.cpu().tolist()
    counts=output_lengths.cpu().tolist()
    return [row[:length] for row,length in zip(tokens,counts)]


@torch.no_grad()
def infer_batch(model, adapter, tokenizer, targets, sources, pooling, max_new_tokens, stop_flags):
    past,states=context_batch(model,targets,sources,pooling)
    latents=adapter(states)
    logits,layers,lengths=append_latents(model,past,[p.suffix_ids for p in targets],latents)
    return decode_tensor_batch(model,tokenizer,logits,layers,lengths,max_new_tokens,stop_flags)