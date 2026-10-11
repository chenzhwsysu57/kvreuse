import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from scripts import batch_eval as batch
from scripts import run_direct_reuse as base
from kv_semantic_attack.prefix_adapter_repeat import (PrefixLatentAdapter,
    target_prefix_states_batch,append_dynamic_latents_batch)
from kv_semantic_attack.universal_soft_suffix import freeze_model
from kv_semantic_attack.tensor_adapter_inference import context_batch,append_latents,decode_tensor_batch,infer_batch


class TensorInferenceTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(23)
        cfg=Qwen3Config(vocab_size=40,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=128,attention_dropout=0,bos_token_id=None,eos_token_id=None,pad_token_id=0)
        model=Qwen3ForCausalLM(cfg).eval();freeze_model(model)
        return model

    def parts(self):
        def p(prefix,block,suffix):
            return SimpleNamespace(full_ids=torch.tensor(prefix+block+suffix),prefix_ids=torch.tensor(prefix),
                block_ids=torch.tensor(block),suffix_ids=torch.tensor(suffix),block_token_start=len(prefix),
                block_token_end=len(prefix)+len(block))
        a=p([1,2,3,4],[5,6],[7]);b=p([8],[5,6],[9,10,11])
        c=p([12,13],[14],[15,16]);d=p([17],[14],[18])
        return [a,b,c,d],[b,a,d,c]

    def test_context_states_kv_and_prefill_match_legacy(self):
        model=self.model();targets,sources=self.parts()
        adapter=PrefixLatentAdapter(32,3,8)
        tokenizer=SimpleNamespace(eos_token_id=None,encode=lambda *a,**kw:[39])
        with torch.no_grad():
            _,source_layers=batch.prefill_ids_batch(model,[p.full_ids for p in sources])
            _,prefix_layers=batch.prefill_ids_batch(model,[p.prefix_ids for p in targets])
            old=[]
            for t,s,donor,prefix in zip(targets,sources,source_layers,prefix_layers):
                block=base.slice_cache(donor,s.block_token_start,s.block_token_end)
                old.append(base.splice_prefix_block(prefix,base.relocate_block(model,block,s.block_token_start,t.block_token_start)))
            for pooling in ('last','mean'):
                calls=[]
                hook=model.model.register_forward_pre_hook(lambda m,a,kw:calls.append(kw['input_ids'].shape),with_kwargs=True)
                past,states=context_batch(model,targets,sources,pooling)
                hook.remove();self.assertEqual(len(calls),1)
                expected_states=target_prefix_states_batch(model,[p.prefix_ids for p in targets],pooling)
                torch.testing.assert_close(states,expected_states,rtol=1e-5,atol=1e-5)
                expected=batch.pad_cache(list(zip(old,past.lengths)),max(past.lengths))
                for pair,want in zip(past.layers,expected):
                    for x,y in zip(pair,want):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)
                latent=adapter(states)
                old_logits,old_cache,old_lengths=append_dynamic_latents_batch(model,old,[p.suffix_ids for p in targets],latent)
                logits,layers,lengths=append_latents(model,past,[p.suffix_ids for p in targets],latent)
                self.assertEqual(lengths,old_lengths)
                torch.testing.assert_close(logits,old_logits,rtol=1e-5,atol=1e-5)
                padded=batch.pad_cache(list(zip(old_cache,old_lengths)),max(old_lengths))
                for pair,want in zip(layers,padded):
                    for x,y in zip(pair,want):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-5)
                new_tokens=decode_tensor_batch(model,tokenizer,logits,layers,lengths,5,[False]*4)
                old_tokens=batch.decode_batch(model,tokenizer,old_logits,padded,old_lengths,5,[False]*4)
                self.assertEqual(new_tokens,old_tokens)

    def test_batch_matches_serial_odd_tail(self):
        model=self.model();targets,sources=self.parts();targets=targets[:3];sources=sources[:3]
        adapter=PrefixLatentAdapter(32,3,8)
        tok=SimpleNamespace(eos_token_id=None,encode=lambda *a,**kw:[39])
        calls=[]
        def capture(module,a,kw):
            tensor=kw.get('input_ids')
            if tensor is None:tensor=kw['inputs_embeds']
            calls.append(tuple(tensor.shape))
        hook=model.model.register_forward_pre_hook(capture,with_kwargs=True)
        actual=infer_batch(model,adapter,tok,targets,sources,'last',5,[False]*3)
        hook.remove()
        self.assertEqual(len(calls),2+max(map(len,actual))-1)
        self.assertEqual(calls[1][0],3)
        self.assertTrue(all(shape==(3,1) for shape in calls[2:]))
        for i in range(3):
            expected=infer_batch(model,adapter,tok,[targets[i]],[sources[i]],'last',5,[False])
            self.assertEqual(actual[i],expected[0])

    def test_stop_masks_multiple_eos_brace_and_output_lengths(self):
        class Decoder:
            def __init__(self):self.calls=0
            def __call__(self,**kw):
                self.calls+=1
                token=torch.tensor([4,6,7] if self.calls==1 else [4,4,8])
                hidden=torch.nn.functional.one_hot(token,10).float()[:,None,:]
                return SimpleNamespace(last_hidden_state=hidden,past_key_values=kw['past_key_values'])
        decoder=Decoder()
        model=SimpleNamespace(model=decoder,lm_head=lambda x:x)
        tokenizer=SimpleNamespace(eos_token_id=[6,8],encode=lambda *a,**kw:[4])
        layers=[(torch.zeros(3,1,4,2),torch.zeros(3,1,4,2))]
        logits=torch.nn.functional.one_hot(torch.tensor([4,5,5]),10).float()
        with patch('kv_semantic_attack.tensor_adapter_inference.batch_ops.make_batch_cache',side_effect=lambda m,c:c):
            tokens=decode_tensor_batch(model,tokenizer,logits,layers,[2,4,3],10,[True,False,False])
        self.assertEqual(tokens,[[4],[5,6],[5,7,8]])
        self.assertEqual(decoder.calls,2)


if __name__=='__main__':unittest.main()