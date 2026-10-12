import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from latent_remedy.tensor_batch import project_spans, PackedLogits
from latent_remedy.prefix_adapter_repeat import PrefixLatentAdapter
from latent_remedy.paired_full_context import PairedFullContext
from latent_remedy.train_prefix_adapter_repeat import (
    kl, masked_answer_kd, masked_gold_first_token_ce)
from latent_remedy.universal_soft_suffix import continuation_logits_batch
from latent_remedy.paired_tensor_training import backward_paired_batch, packed_mixed_loss
from latent_remedy.prepared_paired_tokens import PreparedPairedTokens
from scripts import batch_eval


class CountingHead(torch.nn.Linear):
    def __init__(self, hidden, vocab):
        super().__init__(hidden,vocab,bias=False)
        self.calls=0

    def forward(self, x):
        self.calls+=1
        return super().forward(x)


class TensorTrainingTests(unittest.TestCase):
    def test_adapter_batched_values_and_gradients(self):
        torch.manual_seed(3)
        adapter=PrefixLatentAdapter(8,3,5)
        torch.nn.init.normal_(adapter.network[-1].weight)
        states=torch.randn(4,8)
        expected=torch.stack([adapter(s) for s in states])
        actual=adapter(states)
        torch.testing.assert_close(actual,expected)
        actual.sum().backward()
        self.assertIsNotNone(adapter.network[-1].weight.grad)

    def test_packed_projection_one_call_variable_lengths(self):
        model=SimpleNamespace(lm_head=CountingHead(8,11))
        hidden=torch.randn(3,7,8,requires_grad=True)
        actual=project_spans(model,hidden,[2,0],[1,3],[4,1])
        self.assertEqual(model.lm_head.calls,1)
        torch.testing.assert_close(actual[0],hidden[2,1:5]@model.lm_head.weight.T)
        torch.testing.assert_close(actual[1],hidden[0,3:4]@model.lm_head.weight.T)
        sum(x.sum() for x in actual).backward()
        self.assertTrue(torch.count_nonzero(hidden.grad[2,1:5])>0)
        self.assertEqual(torch.count_nonzero(hidden.grad[1]),0)

    def test_kd_preserves_mean_per_example_and_gradients(self):
        torch.manual_seed(4)
        teacher=[torch.randn(n,13) for n in (1,5,2,3)]
        student=[torch.randn(n,13,requires_grad=True) for n in (1,5,2,3)]
        indices=[1,3]
        expected=torch.stack([kl(teacher[i],student[i],1.7) for i in indices]).mean()
        actual=masked_answer_kd(teacher,student,indices,1.7)
        torch.testing.assert_close(actual,expected)
        expected_grad=torch.autograd.grad(expected,[student[i] for i in indices],retain_graph=True)
        actual_grad=torch.autograd.grad(actual,[student[i] for i in indices])
        for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a,b)
        self.assertEqual(float(masked_answer_kd(teacher,student,[],1).detach()),0)

    def test_ce_only_first_token_selected_rows(self):
        student=[torch.randn(n,13,requires_grad=True) for n in (2,5,1)]
        traces=[torch.zeros(n,dtype=torch.long) for n in (2,5,1)]
        expected=torch.stack([torch.nn.functional.cross_entropy(student[i][:1],traces[i][:1])
                              for i in (0,2)]).mean()
        actual=masked_gold_first_token_ce(student,traces,[0,2])
        torch.testing.assert_close(actual,expected)
        actual.backward()
        self.assertIsNone(student[1].grad)
        self.assertEqual(torch.count_nonzero(student[0].grad[1]),0)

    def test_packed_loss_matches_list_values_and_gradients(self):
        lengths=[2,5,1,3]
        logits=torch.randn(sum(lengths),13,requires_grad=True)
        students=list(logits.split(lengths))
        teachers=[None,torch.randn(5,13),None,torch.randn(3,13)]
        packed_teacher=PackedLogits(torch.cat([teachers[1],teachers[3]]),[5,3],[1,3])
        answers=[torch.zeros(n,dtype=torch.long) for n in lengths]
        for kd_indices,gold_indices in [([1,3],[0,2]),([3],[0,2]),([],[]),([1,3],[])]:
            kd,ce=packed_mixed_loss(packed_teacher,PackedLogits(logits,lengths,[0,1,2,3]),
                                    answers,kd_indices,gold_indices,1.3)
            expected_kd=masked_answer_kd(teachers,students,kd_indices,1.3)
            expected_ce=masked_gold_first_token_ce(students,answers,gold_indices)
            torch.testing.assert_close(kd,expected_kd)
            torch.testing.assert_close(ce,expected_ce)
            new_grad=torch.autograd.grad(kd+ce,logits,retain_graph=True)[0]
            old_grad=torch.autograd.grad(expected_kd+expected_ce,logits,retain_graph=True)[0]
            torch.testing.assert_close(new_grad,old_grad)

    def test_per_task_audit_values_and_retry_reset(self):
        lengths=[1,2,1,3]
        student=PackedLogits(torch.randn(7,9,requires_grad=True),lengths,[0,1,2,3])
        teachers=[torch.randn(2,9),torch.randn(3,9)]
        teacher=PackedLogits(torch.cat(teachers),[2,3],[1,3])
        answers=[torch.zeros(n,dtype=torch.long) for n in lengths]
        audit={}
        kd,ce=packed_mixed_loss(teacher,student,answers,[1,3],[0,2],1,audit=audit)
        torch.testing.assert_close(audit['kd'].mean(),kd.detach())
        torch.testing.assert_close(audit['ce'].mean(),ce.detach())
        parameter=torch.nn.Parameter(torch.tensor(1.0));records=[]
        def compute(chunk):
            records.extend(chunk)
            if len(chunk)>4:raise torch.OutOfMemoryError('test')
            return {'loss':parameter.square()}
        backward_paired_batch(list(range(8)),compute,[parameter],reset_audit=records.clear)
        self.assertEqual(records,list(range(8)))

    def test_prepared_tables_reordered_pairs_and_tail_without_repacking(self):
        prompts={};traces={}
        for task in ('long','short'):
            for side in ('a','b'):
                for mode in (False,True):
                    prefix=[1,2] if side=='a' else [3]
                    block=[4,5] if task=='long' else [4]
                    suffix=[6,7] if mode else [8]
                    entry=(task,side,mode)
                    prompts[entry]=SimpleNamespace(full_ids=torch.tensor(prefix+block+suffix),
                        prefix_ids=torch.tensor(prefix),block_ids=torch.tensor(block),suffix_ids=torch.tensor(suffix),
                        block_token_start=len(prefix),block_token_end=len(prefix)+len(block))
                    traces[entry]=torch.tensor([9,10,11]) if mode else torch.tensor([12])
        store=PreparedPairedTokens(prompts,traces,'cpu')
        batch=[({'task_id':task},side,mode) for task in ('short','long')
               for side in ('a','b') for mode in (False,True)]
        for selected in (batch,batch[:4]):
            with patch('torch.cat',side_effect=AssertionError('per-step token cat')), \
                 patch('torch.equal',side_effect=AssertionError('per-step block validation')):
                prepared=store.batch(selected)
            rows=[i for i,(_,_,mode) in enumerate(selected) if mode]
            for row,i in enumerate(rows):
                r,s,m=selected[i];entry=(r['task_id'],s,m)
                expected=torch.cat((prompts[entry].full_ids,traces[entry][:-1]))
                n=prepared['teacher']['sequence_lengths'][row]
                torch.testing.assert_close(prepared['teacher']['ids'][row,-n:],expected)
                self.assertEqual(torch.count_nonzero(prepared['teacher']['ids'][row,:-n]),0)
            for row,(r,s,m) in enumerate(selected):
                entry=(r['task_id'],s,m)
                expected=torch.cat((prompts[entry].suffix_ids,traces[entry][:-1]))
                torch.testing.assert_close(prepared['student']['ids'][row,:len(expected)],expected)

    def test_optimizer_zero_grad_and_normal_single_compute(self):
        parameter=torch.nn.Parameter(torch.tensor(2.0))
        optimizer=torch.optim.SGD([parameter],lr=.1)
        calls=[]
        def compute(chunk):
            calls.append(len(chunk));return {'loss':parameter.square()}
        with patch.object(optimizer,'zero_grad',wraps=optimizer.zero_grad) as zero:
            result,cap,retries=backward_paired_batch(list(range(32)),compute,[parameter],32,optimizer=optimizer)
        self.assertEqual(calls,[32]);self.assertEqual(cap,32);self.assertEqual(retries,0)
        zero.assert_called_once_with(set_to_none=True)
        torch.testing.assert_close(parameter.grad,torch.tensor(4.0))

    def test_oom_retry_discards_partial_gradient_preserves_effective_batch(self):
        parameter=torch.nn.Parameter(torch.tensor(2.0))
        batch=list(range(16))
        attempts=[]
        def compute(chunk):
            attempts.append(tuple(chunk))
            # Fail after a partial successful backward at cap=8 too.
            if len(chunk)>8 or (len(chunk)==8 and chunk[0]==8):
                raise torch.OutOfMemoryError('simulated OOM')
            values=torch.tensor(chunk,dtype=torch.float)
            loss=((parameter-values)**2).mean()+1e-4*parameter.square()
            return {'loss':loss,'answer_loss':loss}
        totals,cap,retries=backward_paired_batch(batch,compute,[parameter])
        self.assertEqual(cap,4);self.assertEqual(retries,2)
        expected=((parameter-torch.arange(16).float())**2).mean()+1e-4*parameter.square()
        expected_gradient=torch.autograd.grad(expected,parameter)[0]
        torch.testing.assert_close(parameter.grad,expected_gradient)
        torch.testing.assert_close(totals['loss'],expected.detach())
        self.assertIn(tuple(range(8)),attempts)

    def test_oom_minimum_pair_raises_without_partial_gradient(self):
        parameter=torch.nn.Parameter(torch.tensor(2.0))
        def fail(chunk):raise torch.OutOfMemoryError('minimum pair does not fit')
        with self.assertRaises(torch.OutOfMemoryError):
            backward_paired_batch(list(range(4)),fail,[parameter])
        self.assertIsNone(parameter.grad)

    def test_tensor_kv_matches_role_reference_variable_lengths_rope(self):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.calls=0
            def forward(self,input_ids,**kwargs):
                self.calls+=1
                h=torch.stack((input_ids.float(),input_ids.float()+1),dim=-1)
                return SimpleNamespace(last_hidden_state=h,past_key_values=[(h[:,None],h[:,None])]*2)
        decoder=Decoder().eval()
        model=SimpleNamespace(training=False,parameters=lambda:[],model=decoder,lm_head=CountingHead(2,7))
        def prompt(prefix,block):
            ids=torch.tensor(prefix+block+[9])
            return SimpleNamespace(full_ids=ids,prefix_ids=torch.tensor(prefix),block_ids=torch.tensor(block),
                block_token_start=len(prefix),block_token_end=len(prefix)+len(block),suffix_ids=torch.tensor([9]))
        a,b=prompt([1],[7,8]),prompt([2,3,4],[7,8])
        c,d=prompt([5,6],[11]),prompt([12],[11])
        targets=[a,a,b,b,c,c,d,d];sources=[b,b,a,a,d,d,c,c]
        traces=[torch.tensor([3,4,5]) if i%2 else torch.tensor([3]) for i in range(8)]
        modes=[bool(i%2) for i in range(8)]
        entries=[(str(i//4),'a' if i%4<2 else 'b',modes[i]) for i in range(8)]
        token_store=PreparedPairedTokens(dict(zip(entries,targets)),dict(zip(entries,traces)),'cpu')
        batch=[({'task_id':task},side,mode) for task,side,mode in entries]
        prepared=token_store.batch(batch)
        prepared['teacher']['packed']=False
        def rope(model,reference,positions):
            angle=positions.float()[None,None,:,None].expand(1,1,-1,2)/10
            return angle.cos(),angle.sin()
        with (
            patch('latent_remedy.paired_full_context.base.input_device', return_value=torch.device('cpu')),
            patch('latent_remedy.paired_full_context.base.extract_cache_tensors', side_effect=lambda c: c),
            patch('scripts.run_direct_reuse.rope_cos_sin', side_effect=rope),
        ):
            old,old_states,old_teacher=PairedFullContext(model,max_bytes=0).get_batch(targets,sources,traces,modes)
            decoder.calls=0;model.lm_head.calls=0
            new,states,teacher=PairedFullContext(model,max_bytes=0).get_batch(targets,sources,traces,modes,tensorized=True)
            prepared_past,prepared_states,prepared_teacher=PairedFullContext(model,max_bytes=0).get_batch(
                targets,sources,traces,modes,tensorized=True,prepared_tokens=prepared['teacher'])
        self.assertEqual(decoder.calls,2);self.assertEqual(model.lm_head.calls,2)
        torch.testing.assert_close(prepared_states,states)
        for pair,ref in zip(prepared_past.layers,new.layers):
            for actual,want in zip(pair,ref):torch.testing.assert_close(actual,want)
        for actual,want in zip(prepared_teacher,teacher):
            if want is None:self.assertIsNone(actual)
            else:torch.testing.assert_close(actual,want)
        torch.testing.assert_close(states,old_states)
        expected=batch_eval.pad_cache(list(zip(old,new.lengths)),max(new.lengths))
        for pair,ref in zip(new.layers,expected):
            for actual,want in zip(pair,ref):torch.testing.assert_close(actual,want)
        for actual,want in zip(teacher,old_teacher):
            if want is None:self.assertIsNone(actual)
            else:torch.testing.assert_close(actual,want)

    def test_student_one_embedding_projection_and_gradient(self):
        class Decoder(torch.nn.Module):
            def forward(self,inputs_embeds,**kwargs):
                self.inputs=inputs_embeds;self.kwargs=kwargs
                return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1))
        embedding=torch.nn.Embedding(20,4)
        decoder=Decoder()
        model=SimpleNamespace(model=decoder,lm_head=CountingHead(4,20),get_input_embeddings=lambda:embedding)
        latents=torch.randn(2,3,4,requires_grad=True)
        pasts=[[(torch.zeros(1,1,n,2),torch.zeros(1,1,n,2))] for n in (2,4)]
        suffixes=[torch.tensor([3,4]),torch.tensor([5])]
        answers=[torch.tensor([6,7,8]),torch.tensor([9])]
        with patch('latent_remedy.universal_soft_suffix.base.input_device',return_value=torch.device('cpu')), \
             patch('scripts.batch_eval.make_batch_cache',side_effect=lambda m,p:p), \
             patch.object(embedding,'forward',wraps=embedding.forward) as embed:
            result=continuation_logits_batch(model,list(zip(pasts,suffixes,answers)),latents)
            result_prepared=continuation_logits_batch(model,list(zip(pasts,suffixes,answers)),latents,
                prepared_tokens={'ids':torch.tensor([[3,4,6,7],[5,0,0,0]]),
                                 'suffix_lengths':[2,1],'answer_lengths':[3,1]})
        self.assertEqual(embed.call_count,2);self.assertEqual(model.lm_head.calls,2)
        for actual,want in zip(result_prepared,result):torch.testing.assert_close(actual,want)
        old_grad=torch.autograd.grad(sum(x.sum() for x in result),latents,retain_graph=True)[0]
        new_grad=torch.autograd.grad(sum(x.sum() for x in result_prepared),latents,retain_graph=True)[0]
        torch.testing.assert_close(old_grad,new_grad)
        self.assertEqual([len(x) for x in result],[3,1])
        self.assertEqual(decoder.kwargs['position_ids'].tolist(),[[2,3,4,5,6,7,8],[4,5,6,7,0,0,0]])
        sum(x.sum() for x in result).backward()
        self.assertTrue(torch.count_nonzero(latents.grad)>0)


if __name__=='__main__':unittest.main()