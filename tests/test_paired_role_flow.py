import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from kv_semantic_attack.paired_full_context import PairedFullContext
from kv_semantic_attack.train_prefix_adapter_repeat import paired_role_epoch_batches


class PairedFlowTests(unittest.TestCase):
    def test_full_epoch_pair_coverage(self):
        pools={'small':[], 'large':[]}
        for family,n in [('small',3),('large',15)]:
            for i in range(n):
                for side in ('a','b'):
                    for mode in (False,True):pools[family].append(({'task_id':f'{family}-{i}'},side,mode))
        batches=paired_role_epoch_batches(pools,1)
        self.assertEqual(len(batches),18)
        for _,batch in batches:
            self.assertEqual(len({r['task_id'] for r,_,_ in batch}),1)
            self.assertEqual({(s,m) for _,s,m in batch},{('a',False),('a',True),('b',False),('b',True)})
            large=paired_role_epoch_batches(pools,1,16)
            self.assertEqual(len(large),5)
            self.assertEqual([len(b) for _,b in large],[16,16,16,16,8])
            self.assertEqual(sorted((r['task_id'],s,m) for _,b in batches for r,s,m in b),
                     sorted((r['task_id'],s,m) for _,b in large for r,s,m in b))

    def test_one_dense_call_swaps_blocks_and_no_retained_cache(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.calls=0
            def forward(self, input_ids, **kwargs):
                self.calls+=1
                values=input_ids.float()[...,None]
                return SimpleNamespace(last_hidden_state=values,
                    past_key_values=[(values[:,None],values[:,None])])
        decoder=Model().eval()
        model=SimpleNamespace(training=False,parameters=lambda:[],model=decoder,lm_head=lambda h:h)
        def prompt(role):
            return SimpleNamespace(full_ids=torch.tensor([role,7,8,9]),prefix_ids=torch.tensor([role]),
                block_ids=torch.tensor([7,8]),block_token_start=1,block_token_end=3)
        a,b=prompt(1),prompt(2)
        engine=PairedFullContext(model,max_bytes=0)
        with patch('kv_semantic_attack.paired_full_context.base.input_device',return_value=torch.device('cpu')), \
             patch('kv_semantic_attack.paired_full_context.base.extract_cache_tensors',side_effect=lambda c:c):
            pasts,states,teacher=engine.get_batch([a,a,b,b],[b,b,a,a],
                [torch.tensor([3,4])]*4,[False,True,False,True])
        self.assertEqual(decoder.calls,1)
        self.assertEqual(engine.bytes,0);self.assertEqual(len(engine.cache),0)
        self.assertEqual(states.flatten().tolist(),[1,1,2,2])
        self.assertEqual(pasts[0][0][0].flatten().tolist(),[1,7,8])
        self.assertIsNone(teacher[0]);self.assertIsNone(teacher[2])
        self.assertEqual(teacher[1].flatten().tolist(),[9,3])

if __name__=='__main__':unittest.main()