"""Token-only GPU tables; no dataset KV/logit cache or per-step token copies."""
import torch


class PreparedPairedTokens:
    def __init__(self, prompts, traces, device):
        self.entries = list(prompts)
        self.index = {entry:i for i,entry in enumerate(self.entries)}
        self.context_ids = {}
        context_keys = {}
        self.suffix_lengths = []
        self.answer_lengths = []
        self.full_lengths = []
        self.prefix_lengths = []
        self.block_starts = []
        self.block_ends = []
        student, teacher = [], []
        for entry in self.entries:
            p = prompts[entry]
            trace = traces[entry].detach().cpu()
            # One-time CPU preparation; only padded integer tables go to GPU.
            student.append(torch.cat((p.suffix_ids.cpu(),trace[:-1])))
            teacher.append(torch.cat((p.full_ids.cpu(),trace[:-1])) if entry[2] else p.full_ids.cpu())
            self.suffix_lengths.append(len(p.suffix_ids))
            self.answer_lengths.append(len(trace))
            self.full_lengths.append(len(p.full_ids))
            self.prefix_lengths.append(len(p.prefix_ids))
            self.block_starts.append(p.block_token_start)
            self.block_ends.append(p.block_token_end)
            key = tuple(p.full_ids[:p.block_token_end].tolist())
            self.context_ids[entry] = context_keys.setdefault(key,len(context_keys))
            donor = prompts[(entry[0],'b' if entry[1]=='a' else 'a',entry[2])]
            if not torch.equal(p.block_ids.cpu(),donor.block_ids.cpu()):
                raise ValueError('shared block token mismatch')
        self.teacher_lengths = [len(seq) for seq in teacher]
        self.student_lengths = [len(seq) for seq in student]
        self.teacher = torch.nn.utils.rnn.pad_sequence(teacher,batch_first=True).to(device)
        self.student = torch.nn.utils.rnn.pad_sequence(student,batch_first=True).to(device)
        self.device = device
        self.bytes = sum(t.numel()*t.element_size() for t in (self.teacher,self.student))

    def batch(self, batch):
        entries = [(r['task_id'],s,m) for r,s,m in batch]
        indices = [self.index[e] for e in entries]
        # These loops handle CPU IDs only, not token tensors or model calls.
        request = [i for i,e in zip(indices,entries) if e[2]]
        lookup = {self.context_ids[e]:row for row,e in enumerate(e for e in entries if e[2])}
        spans = [(row,self.full_lengths[index]-1,self.answer_lengths[index],i)
                 for row,(i,index) in enumerate((i,index) for i,(index,e) in enumerate(zip(indices,entries)) if e[2])]
        donors = [(e[0],'b' if e[1]=='a' else 'a',e[2]) for e in entries]
        if any(self.context_ids[e] not in lookup for e in entries+donors):
            raise ValueError('prepared paired input requires both roles and reasoning modes')
        lengths = [self.teacher_lengths[i] for i in request]
        width = max(lengths)
        row = torch.tensor(request,device=self.device)
        right = self.teacher.index_select(0,row)[:,:width]
        offsets = width-torch.tensor(lengths,device=self.device)
        col = torch.arange(width,device=self.device)
        position = col[None,:]-offsets[:,None]
        ids = right.gather(1,position.clamp_min(0)).masked_fill(position<0,0)
        suffix = [self.suffix_lengths[i] for i in indices]
        answer = [self.answer_lengths[i] for i in indices]
        student_width = max(self.student_lengths[i] for i in indices)
        tokens = self.student.index_select(0,torch.tensor(indices,device=self.device))[:,:student_width]
        return {
            'teacher':{'ids':ids,'sequence_lengths':lengths,'spans':spans,
                       'target_rows':[lookup[self.context_ids[e]] for e in entries],
                       'donor_rows':[lookup[self.context_ids[e]] for e in donors],
                       'prefix_lengths':[self.prefix_lengths[i] for i in indices],
                       'true_lengths':[self.block_ends[i] for i in indices],
                       'source_starts':[self.block_starts[self.index[e]] for e in donors],
                       'target_starts':[self.block_starts[i] for i in indices],
                       'packed':True},
            'student':{'ids':tokens,'suffix_lengths':suffix,'answer_lengths':answer,'packed':True},
        }