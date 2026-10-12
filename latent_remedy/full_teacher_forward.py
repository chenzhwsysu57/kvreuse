"""Teacher-forced answer logits with one dense target-prompt forward."""
import torch
from scripts import run_direct_reuse as base


@torch.no_grad()
def full_teacher_logits_batch(model, targets, traces):
    if len(targets) != len(traces):
        raise ValueError('teacher prompt/trace count mismatch')
    if not targets:
        return []
    sequences = [torch.cat((p.full_ids, trace[:-1])) for p,trace in zip(targets,traces)]
    if any(trace.numel() == 0 for trace in traces):
        raise ValueError('empty teacher trace')
    width = max(int(s.numel()) for s in sequences)
    device = base.input_device(model)
    ids = torch.zeros((len(sequences),width),device=device,dtype=torch.long)
    mask = torch.zeros_like(ids); positions = torch.zeros_like(ids)
    for i,seq in enumerate(sequences):
        n = int(seq.numel())
        ids[i,-n:] = seq.to(device)
        mask[i,-n:] = 1
        positions[i,-n:] = torch.arange(n,device=device)
    output = model.model(input_ids=ids,attention_mask=mask,position_ids=positions,
                         use_cache=False,return_dict=True)
    results = []
    for i,(seq,prompt,trace) in enumerate(zip(sequences,targets,traces)):
        # Final prompt token predicts answer[0], then teacher-force answer[:-1].
        start = width-int(seq.numel())+int(prompt.full_ids.numel())-1
        hidden = output.last_hidden_state[i,start:start+int(trace.numel())]
        results.append(model.lm_head(hidden).float())
    return results