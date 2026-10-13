"""Original precompute/read-only-states pattern for prefix-conditioned adapters."""
import torch
import time
import resource
import json


def to_host(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, tuple):
        return tuple(to_host(v) for v in value)
    if isinstance(value, list):
        return [to_host(v) for v in value]
    return value


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, copy=True)
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    return value


@torch.no_grad()
def reference_states(trainer, model, tokenizer, batch, trace_cache, teacher_tokens, pooling):
    targets = [trainer.parts(tokenizer,r,s,m) for r,s,m in batch]
    sources = [trainer.parts(tokenizer,r,'b' if s=='a' else 'a',m) for r,s,m in batch]
    pasts = trainer.reuse_pasts_batch(model,targets,sources)
    tp,ts = trainer.full_teacher_pasts_batch(model,targets)
    traces = trainer.mixed_traces_batch(model,tokenizer,batch,tp,ts,teacher_tokens,
                                       trace_cache=trace_cache)
    teacher = trainer.continuation_logits_batch(model,list(zip(tp,ts,traces)),None)
    states = trainer.target_prefix_states_batch(model,[p.prefix_ids for p in targets],pooling)
    # Full logits are needed only for reasoning KL, not direct gold CE.
    return [(pasts[i], targets[i].suffix_ids, traces[i], states[i], teacher[i] if m else None)
            for i,(_,_,m) in enumerate(batch)]


def load_batch(cache, batch, device):
    values = [to_device(cache[(r['task_id'],s,m)], device) for r,s,m in batch]
    pasts,suffixes,traces,states,teacher = zip(*values)
    return list(pasts),list(suffixes),list(traces),torch.stack(states),list(teacher)


@torch.no_grad()
def precompute(trainer, model, tokenizer, pools, args, trace_cache):
    from tqdm.auto import tqdm
    batches = trainer.complete_epoch_batches(pools,args.train_batch_size,args.seed)
    cache = {}
    started = time.perf_counter()
    tensor_bytes = 0
    def size(value):
        if isinstance(value,torch.Tensor):
            return value.numel()*value.element_size()
        if isinstance(value,(tuple,list)):
            return sum(size(v) for v in value)
        return 0
    for _,batch in tqdm(batches,desc='cache frozen training states',unit='batch'):
        rows = reference_states(trainer,model,tokenizer,batch,trace_cache,
                                args.teacher_tokens,args.prefix_pooling)
        for (r,s,m),row in zip(batch,rows):
            key=(r['task_id'],s,m)
            if key in cache:
                raise ValueError('duplicate frozen training direction')
            cache[key]=to_host(row)
            tensor_bytes += size(cache[key])
    expected=sum(len(v) for v in pools.values())
    if len(cache)!=expected:
        raise ValueError('incomplete frozen training states')
    print(f'[static-states] {len(cache)} cached directions; epochs call student forward only',flush=True)
    trainer.base.write_json_atomic(args.output/'static_precompute_metrics.json',
        {'directions':len(cache),'wall_seconds':time.perf_counter()-started,
         'tensor_bytes':tensor_bytes,'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'storage':'CPU tensors, copied to GPU each batch', 'complete':True})
    return cache