import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from kv_semantic_attack.full_teacher_forward import full_teacher_logits_batch


class FullTeacherTests(unittest.TestCase):
    def test_one_forward_correct_answer_offsets_and_mask(self):
        calls=[]
        def decoder(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(last_hidden_state=kwargs['input_ids'].float().unsqueeze(-1))
        model=SimpleNamespace(model=decoder,lm_head=lambda x:x)
        prompts=[SimpleNamespace(full_ids=torch.tensor([1,2,3])),
                 SimpleNamespace(full_ids=torch.tensor([4,5]))]
        traces=[torch.tensor([6,7,8]),torch.tensor([9])]
        with patch('kv_semantic_attack.full_teacher_forward.base.input_device',return_value=torch.device('cpu')):
            values=full_teacher_logits_batch(model,prompts,traces)
        self.assertEqual(len(calls),1)
        self.assertEqual([v.flatten().tolist() for v in values],[[3.,6.,7.],[5.]])
        self.assertEqual(calls[0]['attention_mask'].tolist(),[[1,1,1,1,1],[0,0,0,1,1]])
        self.assertFalse(calls[0]['use_cache'])


if __name__=='__main__':unittest.main()