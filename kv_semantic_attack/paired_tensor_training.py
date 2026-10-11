"""Tensorized paired updates with OOM-safe, effective-batch-preserving retry."""
import gc
import torch
from kv_semantic_attack.universal_soft_suffix import continuation_logits_batch
from kv_semantic_attack.tensor_batch import PackedLogits


def packed_mixed_loss(teacher, student, answers, kd_indices, gold_indices, temperature, audit=None):
    """Keep projected logits packed: no split into rows then cat them again."""
    device = student.logits.device
    counts = torch.tensor(student.lengths,device=device)
    offsets = torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
    zero = student.logits.reshape(-1)[0]*0.0
    if kd_indices:
        row_mask = torch.zeros(len(counts),device=device,dtype=torch.bool)
        row_mask[torch.tensor(kd_indices,device=device)] = True
        token_mask = torch.repeat_interleave(row_mask,counts)
        prediction = student.logits[token_mask]
        teacher_mask = torch.tensor([i in kd_indices for i in teacher.example_ids],device=device)
        reference = teacher.logits[torch.repeat_interleave(teacher_mask,torch.tensor(teacher.lengths,device=device))]
        if prediction.shape != reference.shape:
            raise ValueError('packed teacher/student spans differ')
        selected_counts = counts[torch.tensor(kd_indices,device=device)]
        weights = torch.repeat_interleave(selected_counts.float().reciprocal(),selected_counts)
        per_token = torch.nn.functional.kl_div(
            torch.log_softmax(prediction/temperature,-1),torch.softmax(reference/temperature,-1),
            reduction='none').sum(-1)
        kd = (per_token*weights).sum()*(temperature**2/len(kd_indices))
        if audit is not None:
            token_rows=torch.repeat_interleave(torch.arange(len(selected_counts),device=device),selected_counts)
            audit['kd']=torch.zeros(len(selected_counts),device=device).scatter_add_(0,token_rows,
                per_token.detach()*weights*temperature**2)
    else:
        kd = zero
    if gold_indices:
        indices = torch.tensor(gold_indices,device=device)
        labels = torch.stack([answers[i][0] for i in gold_indices])
        ce = torch.nn.functional.cross_entropy(student.logits.index_select(0,offsets[indices]),labels)
        if audit is not None:
            audit['ce']=torch.nn.functional.cross_entropy(student.logits.index_select(0,offsets[indices]),
                                                          labels,reduction='none').detach()
    else:
        ce = zero
    return kd,ce


def backward_paired_batch(batch, compute_loss, parameters, microbatch_size=None, *, optimizer=None, reset_audit=None):
    """Retry the *entire* update, before optimizer.step, at smaller pair chunks.

    Each chunk contains whole A/B x direct/reasoning pairs. Weight its
    per-mode mean and regularizer by chunk_size/effective_batch_size. This
    preserves the objective and update count; only memory microbatching changes.
    The callable includes teacher, student, loss and all their temporary tensors.
    """
    if len(batch) % 4:
        raise ValueError('paired effective batch must contain complete four-row pairs')
    parameters = list(parameters)
    cap = min(microbatch_size or len(batch), len(batch))
    if cap < 4 or cap % 4:
        raise ValueError('paired microbatch must be a positive multiple of four')
    retries = 0

    def zero_grad():
        if reset_audit is not None:reset_audit()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            for parameter in parameters:
                parameter.grad = None

    def attempt():
        if cap == len(batch):
            values = compute_loss(batch)
            values['loss'].backward()
            return {key:value.detach() for key,value in values.items()}
        totals = None
        for start in range(0,len(batch),cap):
            chunk = batch[start:start+cap]
            values = compute_loss(chunk)
            weight = len(chunk)/len(batch)
            (values['loss']*weight).backward()
            detached = {key:value.detach()*weight for key,value in values.items()}
            totals = detached if totals is None else {key:totals[key]+value for key,value in detached.items()}
            del values
        return totals

    while True:
        zero_grad()
        try:
            totals = attempt()
            return totals, cap, retries
        except torch.OutOfMemoryError:
            zero_grad()
            if cap == 4:
                raise
        # Outside the except block: release the failed attempt's traceback too.
        cap = max(4,(cap//2//4)*4)
        retries += 1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def paired_loss(model, adapter, context, batch, prompts, traces, args, kd_directions,
                kd_loss, ce_loss, token_store=None, audit_rows=None):
    target = [prompts[(record['task_id'],side,mode)] for record,side,mode in batch]
    source = [prompts[(record['task_id'],'b' if side=='a' else 'a',mode)]
              for record,side,mode in batch]
    answers = [traces[(record['task_id'],side,mode)] for record,side,mode in batch]
    prepared = token_store.batch(batch) if token_store is not None else None
    context_kwargs = {'prepared_tokens':prepared['teacher']} if prepared is not None else {}
    past, states, teacher = context.get_batch(target,source,answers,
                                             [mode for _,_,mode in batch],tensorized=True,**context_kwargs)
    latent = adapter(states)
    student_kwargs = {'prepared_tokens':prepared['student']} if prepared is not None else {}
    student = continuation_logits_batch(model,
        [(None,p.suffix_ids,answer) for p,answer in zip(target,answers)],latent,batched_past=past,**student_kwargs)
    kd_indices = [i for i,(record,side,mode) in enumerate(batch)
                  if mode and (kd_directions is None or (record['task_id'],side) in kd_directions)]
    gold_indices = [i for i,(_,_,mode) in enumerate(batch) if not mode]
    if isinstance(student,PackedLogits):
        audit={} if audit_rows is not None else None
        answer_loss,gold_loss = packed_mixed_loss(teacher,student,answers,kd_indices,
            gold_indices if args.gold_first_token_ce_weight else [],args.temperature,audit=audit)
        if audit_rows is not None:
            for label,indices in [('kd',kd_indices),('ce',gold_indices)]:
                if label in audit:
                    audit_rows.extend((batch[i][0],batch[i][1],label,value)
                                      for i,value in zip(indices,audit[label].cpu().tolist()))
    else:
        answer_loss = kd_loss(teacher,student,kd_indices,args.temperature)
        gold_loss = ce_loss(student,answers,gold_indices) if args.gold_first_token_ce_weight else answer_loss.new_zeros(())
    loss = (args.kd_weight*answer_loss + args.gold_first_token_ce_weight*gold_loss +
            1e-4*(latent.square().mean()+adapter.base.square().mean()))
    return {'loss':loss,'answer_loss':answer_loss,'gold_ce_loss':gold_loss}