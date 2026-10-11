import unittest
import torch
from kv_semantic_attack.static_adapter_states import to_host, load_batch


class StaticStatesTests(unittest.TestCase):
    def test_exact_constant_roundtrip_and_no_alias(self):
        row = ([(torch.randn(1,2,3,4),torch.randn(1,2,3,4))],
               torch.tensor([1,2]),torch.tensor([3,4]),torch.randn(8),torch.randn(2,9))
        cache = {('x','a',True):to_host(row)}
        batch = [({'task_id':'x'},'a',True)]
        pasts,suffix,traces,states,teacher = load_batch(cache,batch,'cpu')
        torch.testing.assert_close(pasts[0][0][0],row[0][0][0],atol=0,rtol=0)
        torch.testing.assert_close(teacher[0],row[4],atol=0,rtol=0)
        torch.testing.assert_close(states[0],row[3],atol=0,rtol=0)
        pasts[0][0][0].zero_()
        torch.testing.assert_close(cache['x','a',True][0][0][0],row[0][0][0],atol=0,rtol=0)

    def test_direct_has_no_teacher(self):
        cache={('x','b',False):([],torch.tensor([1]),torch.tensor([2]),torch.ones(3),None)}
        values=load_batch(cache,[({'task_id':'x'},'b',False)],'cpu')
        self.assertEqual(values[-1],[None])

if __name__=='__main__':unittest.main()