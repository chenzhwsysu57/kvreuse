import tempfile
import unittest
from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch
import torch
from kv_semantic_attack.teacher_trace_cache import cache_protocol, load_cache, save_cache


class TeacherCacheTests(unittest.TestCase):
    def test_partial_resume_and_complete_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'traces.json'
            protocol = cache_protocol([{'task_id':'x'}], tmp, 64)
            expected = {('x','a'),('x','b')}
            self.assertEqual(load_cache(path,protocol,expected),{})
            partial = {('x','a'):[1,2]}
            save_cache(path,protocol,partial,expected)
            self.assertEqual(load_cache(path,protocol,expected),partial)
            partial[('x','b')] = [3]
            save_cache(path,protocol,partial,expected)
            self.assertEqual(load_cache(path,protocol,expected),partial)
            with self.assertRaises(ValueError):
                load_cache(path,{**protocol,'teacher_tokens':32},expected)
            with self.assertRaises(ValueError):
                load_cache(path,protocol,{('x','a')})

    def test_input_changes_invalidate_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotEqual(cache_protocol([{'prefix':'A'}],tmp,64),
                                cache_protocol([{'prefix':'B'}],tmp,64))

    def test_complete_cache_skips_all_generation_and_records_hit(self):
        from kv_semantic_attack import train_prefix_adapter_repeat as train
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();output=root/'out';output.mkdir()
            args=SimpleNamespace(model_path=model,teacher_tokens=64,teacher_trace_cache=root/'traces.json',
                                 train_batch_size=4,output=output)
            record={'task_id':'x'};pools={'test':[(record,'a',True),(record,'b',True)]}
            expected={('x','a'),('x','b')}
            save_cache(args.teacher_trace_cache,cache_protocol([record],model,64),
                       {('x','a'):[1,2],('x','b'):[3]},expected)
            with patch.object(train,'generate_teachers_batch') as generate, \
                 patch.object(train,'full_teacher_pasts_batch') as prefill:
                result=train.prepare_teacher_trace_cache(None,None,pools,args)
            generate.assert_not_called();prefill.assert_not_called()
            self.assertEqual(len(result),2)
            metrics=json.loads((output/'teacher_cache_metrics.json').read_text())
            self.assertTrue(metrics['cache_hit']);self.assertEqual(metrics['generated_directions'],0)

    def test_partial_cache_generates_only_missing_direction(self):
        from kv_semantic_attack import train_prefix_adapter_repeat as train
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();output=root/'out';output.mkdir()
            args=SimpleNamespace(model_path=model,teacher_tokens=64,teacher_trace_cache=root/'traces.json',
                                 train_batch_size=4,output=output)
            record={'task_id':'x'};pools={'test':[(record,'a',True),(record,'b',True)]}
            expected={('x','a'),('x','b')}
            save_cache(args.teacher_trace_cache,cache_protocol([record],model,64),{('x','a'):[1,2]},expected)
            with patch.object(train,'parts',return_value=object()) as build, \
                 patch.object(train,'full_teacher_pasts_batch',return_value=([None],[None])), \
                 patch.object(train,'generate_teachers_batch',return_value=[torch.tensor([3])]) as generate:
                result=train.prepare_teacher_trace_cache(None,None,pools,args)
            self.assertEqual(build.call_args.args[2],'b');generate.assert_called_once()
            self.assertEqual(result,{('x','a'):[1,2],('x','b'):[3]})
            metrics=json.loads((output/'teacher_cache_metrics.json').read_text())
            self.assertFalse(metrics['cache_hit']);self.assertEqual(metrics['generated_directions'],1)


if __name__ == '__main__':
    unittest.main()