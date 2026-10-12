import unittest
import contextlib
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from latent_remedy import train_prefix_adapter_repeat as train


class MixedSupervisionTests(unittest.TestCase):
    def test_no_reasoning_uses_gold_and_only_reasoning_calls_full(self):
        batch=[({'gold_a':'B'},'a',False),({},'b',True),({'gold_b':'C'},'b',False)]
        tok=SimpleNamespace(encode=lambda text,**kw: {'B}':[1,3],'C}':[2,3]}[text])
        generated=[torch.tensor([0,2])];reference=[torch.zeros(2,4)]
        with patch.object(train.base,'input_device',return_value=torch.device('cpu')), \
             patch.object(train,'full_teacher_pasts_batch',return_value=(['p'],['s'])) as full, \
             patch.object(train,'generate_teachers_batch',return_value=generated), \
             patch.object(train,'continuation_logits_batch',return_value=reference):
            traces,teacher=train.mixed_supervision(None,tok,batch,['a','b','c'],64)
        full.assert_called_once_with(None,['b'])
        self.assertEqual([t.tolist() for t in traces],[[1,3],[0,2],[2,3]])
        self.assertIsNone(teacher[0]);self.assertIsNone(teacher[2])
        self.assertIs(teacher[1],reference[0])

    def test_loss_routes_gradients_and_matches_manual_average(self):
        student=[torch.randn(2,4,requires_grad=True),torch.randn(3,4,requires_grad=True)]
        reference=torch.randn(3,4,requires_grad=True);traces=[torch.tensor([1,3]),torch.tensor([0,2,1])]
        loss,ce,kd=train.mixed_supervision_loss(student,[None,reference],traces,[False,True],1.)
        expected=(torch.nn.functional.cross_entropy(student[0],traces[0])+train.kl(reference.detach(),student[1],1.))/2
        self.assertTrue(torch.allclose(loss,expected));self.assertTrue(torch.allclose(loss,ce+kd))
        loss.backward()
        self.assertIsNone(reference.grad)
        for s in student:self.assertGreater(float(s.grad.abs().sum()),0)
        self.assertGreater(float(student[0].grad[-1].abs().sum()),0)  # full gold trace, not first token only

    def test_single_mode_batches_and_routing_errors(self):
        for reason in (False,True):
            s=torch.randn(2,4,requires_grad=True);teacher=torch.randn(2,4) if reason else None
            loss,ce,kd=train.mixed_supervision_loss([s],[teacher],[torch.tensor([1,3])],[reason],1.)
            self.assertTrue(torch.isfinite(loss));loss.backward()
            self.assertEqual(float(ce if reason else kd),0)
        with self.assertRaises(ValueError):
            train.mixed_supervision_loss([torch.zeros(1,4)],[None],[torch.tensor([1])],[True],1.)


class MixedTraceProtocolTests(unittest.TestCase):
    def test_cached_reasoning_never_generates_and_direct_still_uses_gold(self):
        batch = [({'task_id':'x'}, 'b', True),
                 ({'task_id':'y', 'gold_a':'A'}, 'a', False)]
        tok = SimpleNamespace(encode=Mock(return_value=[1,3]))
        with patch.object(train.base, 'input_device', return_value=torch.device('cpu')), \
             patch.object(train, 'generate_teachers_batch') as generate:
            traces = train.mixed_traces_batch(None, tok, batch, [None,None], [None,None],64,
                                              trace_cache={('x','b'):[4,5,6]})
        generate.assert_not_called()
        self.assertEqual([t.tolist() for t in traces], [[4,5,6],[1,3]])

    def parse(self, *overrides):
        argv = ["train", "--input", "unused.jsonl", "--model-path", "unused-model",
                "--output", "unused-output", "--teacher-trace", "mixed",
                "--teacher-mode", "full", "--train-reasoning", "both",
                "--kd-weight", "1", "--gold-first-token-ce-weight", "1", *overrides]
        with patch.object(sys, "argv", argv):
            return train.parse_args()

    def test_mixed_validation(self):
        args = self.parse()
        self.assertEqual(args.teacher_trace, "mixed")
        self.assertEqual(args.teacher_tokens, 64)
        self.assertEqual(args.gold_first_token_ce_vocab, "full")
        self.assertIsNone(args.init_adapter_checkpoint)
        invalid = [
            ("--teacher-mode", "repeat"), ("--train-reasoning", "no"),
            ("--train-reasoning", "yes"), ("--kd-weight", "0"), ("--kd-weight", "0.5"),
            ("--gold-first-token-ce-weight", "0"), ("--gold-first-token-ce-weight", "2"),
            ("--gold-first-token-ce-vocab", "options"), ("--suffix-kl-weight", "1"),
            ("--teacher-tokens", "65"), ("--teacher-tokens", "0"),
            ("--kd-directions-file", "directions.jsonl"),
            ("--init-adapter-checkpoint", "old.pt"), ("--mixed-gold-full",),
        ]
        for flags in invalid:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    self.parse(*flags)
                self.assertEqual(caught.exception.code, 2)

    def test_legacy_validation_preserved(self):
        self.assertEqual(self.parse("--teacher-trace", "generated", "--teacher-mode", "repeat",
                                   "--gold-first-token-ce-weight", "0").teacher_trace, "generated")
        self.assertEqual(self.parse("--teacher-trace", "gold", "--train-reasoning", "no",
                                   "--gold-first-token-ce-vocab", "options").teacher_trace, "gold")
        self.assertTrue(self.parse("--teacher-trace", "generated", "--mixed-gold-full",
                                  "--gold-first-token-ce-weight", "0").mixed_gold_full)
        for flags in [("--teacher-trace", "gold"), ("--teacher-trace", "generated")]:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse(*flags)

    def test_mixed_trace_selection_preserves_order_and_side(self):
        # Reasoning records deliberately have no gold: they must never reach the gold helper.
        batch = [({}, "a", True), ({"gold_a": "A", "gold_b": "B"}, "b", False),
                 ({}, "b", True), ({"gold_a": "C", "gold_b": "D"}, "a", False)]
        tokenizer = SimpleNamespace(encode=Mock(side_effect=[[1, 3], [2, 3]]))
        pasts = [object() for _ in batch]
        suffixes = [torch.tensor([i]) for i in range(4)]
        generated = [torch.tensor([0, 2, 3]), torch.tensor([3, 0])]
        with patch.object(train.base, "input_device", return_value=torch.device("cpu")), \
             patch.object(train, "generate_teachers_batch", return_value=generated) as generate, \
             patch.object(train, "gold_traces_batch", wraps=train.gold_traces_batch) as gold:
            traces = train.mixed_traces_batch(None, tokenizer, batch, pasts, suffixes, 64)
        self.assertEqual([t.tolist() for t in traces], [[0, 2, 3], [1, 3], [3, 0], [2, 3]])
        self.assertEqual(gold.call_args.args[1], [batch[1], batch[3]])
        self.assertEqual([c.args[0] for c in tokenizer.encode.call_args_list], ["B}", "C}"])
        for call in tokenizer.encode.call_args_list:
            self.assertFalse(call.kwargs["add_special_tokens"])
        generate.assert_called_once_with(None, tokenizer, [pasts[0], pasts[2]],
                                         [suffixes[0], suffixes[2]], 64,
                                         stop_at_closing_brace=[False, False], no_reasoning_tokens=64)
        self.assertIs(traces[0], generated[0])
        self.assertIs(traces[2], generated[1])

    def test_single_mode_traces_skip_unused_helper(self):
        for reasoning in (False, True):
            with self.subTest(reasoning=reasoning):
                batch = [({"gold_a": "B"}, "a", reasoning)]
                tok = SimpleNamespace(encode=Mock(return_value=[1, 3]))
                with patch.object(train.base, "input_device", return_value=torch.device("cpu")), \
                     patch.object(train, "generate_teachers_batch", return_value=[torch.tensor([2])]) as generate, \
                     patch.object(train, "gold_traces_batch", wraps=train.gold_traces_batch) as gold:
                    traces = train.mixed_traces_batch(None, tok, batch, [None], [torch.tensor([0])], 64)
                self.assertEqual(generate.call_count, int(reasoning))
                self.assertEqual(gold.call_count, int(not reasoning))
                self.assertEqual(traces[0].tolist(), [2] if reasoning else [1, 3])

    def test_full_vocab_first_token_ce_and_kd_have_disjoint_gradients(self):
        for modes in ([False, True, False], [True, False, True]):
            with self.subTest(modes=modes):
                student = [torch.linspace(-1, 1, 21).reshape(3, 7).clone().requires_grad_()
                           for _ in modes]
                teacher = [torch.linspace(1, -1, 21).reshape(3, 7).clone().requires_grad_()
                           for _ in modes]
                traces = [torch.tensor([1, 5, 6]) for _ in modes]
                direct = [i for i, reasoning in enumerate(modes) if not reasoning]
                reasoning = [i for i, value in enumerate(modes) if value]
                ce = train.masked_gold_first_token_ce(student, traces, direct)
                kd = train.masked_answer_kd(teacher, student, reasoning, 1.7)
                expected_ce = torch.stack([
                    torch.nn.functional.cross_entropy(student[i][:1], traces[i][:1]) for i in direct
                ]).mean()
                expected_kd = torch.stack([train.kl(teacher[i].detach(), student[i], 1.7)
                                          for i in reasoning]).mean()
                torch.testing.assert_close(ce, expected_ce)
                torch.testing.assert_close(kd, expected_kd)
                ce_grads = torch.autograd.grad(ce, student, retain_graph=True, allow_unused=True)
                kd_grads = torch.autograd.grad(kd, student, retain_graph=True, allow_unused=True)
                for i, is_reasoning in enumerate(modes):
                    if is_reasoning:
                        self.assertIsNone(ce_grads[i])
                        self.assertTrue((kd_grads[i].abs().sum(-1) > 0).all())
                    else:
                        self.assertIsNone(kd_grads[i])
                        self.assertTrue((ce_grads[i][0] != 0).all())  # entire vocab, not options
                        self.assertEqual(ce_grads[i][1:].count_nonzero().item(), 0)
                (ce + kd).backward()
                for reference in teacher:
                    self.assertIsNone(reference.grad)

    def test_empty_masks_are_differentiable_in_single_mode_batches(self):
        for reasoning in (False, True):
            with self.subTest(reasoning=reasoning):
                student = [torch.randn(2, 7, requires_grad=True) for _ in range(2)]
                traces = [torch.tensor([1, 6]) for _ in student]
                teacher = [torch.randn(2, 7, requires_grad=True) for _ in student]
                ce = train.masked_gold_first_token_ce(student, traces, [] if reasoning else [0, 1])
                kd = train.masked_answer_kd(teacher, student, [0, 1] if reasoning else [], 1.)
                empty = ce if reasoning else kd
                self.assertEqual(empty.item(), 0.)
                self.assertTrue(empty.requires_grad)
                zero_grads = torch.autograd.grad(empty, student, retain_graph=True)
                for grad in zero_grads:
                    self.assertEqual(grad.count_nonzero().item(), 0)
                (ce + kd).backward()
                for prediction in student:
                    self.assertGreater(prediction.grad.abs().sum().item(), 0)

    def test_both_prompt_modes_keep_native_thinking_off(self):
        for reasoning in (False, True):
            with self.subTest(reasoning=reasoning), patch.object(train.base, "build_prompt_parts") as build:
                train.parts(None, {}, "a", reasoning)
                build.assert_called_once_with(None, {}, "a", enable_thinking=False,
                                              explicit_reasoning=reasoning, boxed_output=True)

    def test_mock_cpu_training_wires_masks_and_records_protocol(self):
        class TinyAdapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.arange(7, dtype=torch.float32).reshape(1, 7) / 7)

            def forward(self, state):
                return self.base * state[:, None, :]

        # Cover mixed and both single-mode batches through the actual training loop.
        for batch_size, reverse in ((4, False), (1, False), (1, True)):
            with self.subTest(batch_size=batch_size, reverse=reverse), tempfile.TemporaryDirectory() as tmp:
                args = self.parse("--steps", "1", "--train-batch-size", str(batch_size),
                                  "--use-all-records", "--latest-save-interval", "0",
                                  "--output", str(Path(tmp) / "output"))
                record = {"task_id": "toy", "dataset": "toy", "gold_a": "B", "gold_b": "C"}
                tok = SimpleNamespace(encode=lambda text, **kw: {"B}": [1, 6], "C}": [2, 6]}[text])
                model = SimpleNamespace(config=SimpleNamespace(hidden_size=7))
                part = SimpleNamespace(prefix_ids=torch.tensor([0]), suffix_ids=torch.tensor([3]),
                                       full_ids=torch.tensor([0, 3]))
                adapter = TinyAdapter()
                pasts = [object() for _ in range(batch_size)]
                teacher_pasts = [object() for _ in range(batch_size)]
                suffixes = [torch.tensor([3]) for _ in range(batch_size)]
                captured = {}

                def logits(model, contexts, latents):
                    if latents is None:
                        captured["teacher_contexts"] = contexts
                        return [torch.zeros(trace.numel(), 7) for _, _, trace in contexts]
                    predictions = [latents[i].expand(trace.numel(), -1)
                                   for i, (_, _, trace) in enumerate(contexts)]
                    for prediction in predictions:
                        prediction.retain_grad()
                    captured["student"] = predictions
                    captured["traces"] = [trace for _, _, trace in contexts]
                    return predictions

                def generate(model, tokenizer, pasts, suffixes, tokens, **kwargs):
                    self.assertEqual(tokens, 64)
                    self.assertEqual(kwargs["stop_at_closing_brace"], [False] * len(pasts))
                    return [torch.tensor([4, 5, 6]) for _ in pasts]

                progress = Mock()
                progress.__iter__ = Mock(return_value=iter([1]))
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(train, "parse_args", return_value=args))
                    stack.enter_context(patch.object(train, "read_records", return_value=[record]))
                    stack.enter_context(patch.object(train, "task_family", return_value="toy"))
                    stack.enter_context(patch.object(train.random.Random, "shuffle",
                                                    side_effect=lambda pool: pool.reverse() if reverse else None))
                    stack.enter_context(patch.object(train.base, "validate_model_files"))
                    stack.enter_context(patch.object(train.AutoTokenizer, "from_pretrained", return_value=tok))
                    stack.enter_context(patch.object(train.AutoModelForCausalLM, "from_pretrained", return_value=model))
                    stack.enter_context(patch.object(train, "freeze_model"))
                    stack.enter_context(patch.object(train.base, "assert_default_rope"))
                    stack.enter_context(patch.object(train.base, "input_device", return_value=torch.device("cpu")))
                    stack.enter_context(patch.object(train, "PrefixLatentAdapter", return_value=adapter))
                    build = stack.enter_context(patch.object(train.base, "build_prompt_parts", return_value=part))
                    stack.enter_context(patch.object(train.base, "with_repeated_target_prefix", return_value=record))
                    stack.enter_context(patch.object(train, "reuse_pasts_batch", return_value=pasts))
                    full = stack.enter_context(patch.object(train, "full_teacher_contexts_batch",
                                                           return_value=(teacher_pasts, suffixes, None, None)))
                    generated = stack.enter_context(patch.object(train, "generate_teachers_batch", side_effect=generate))
                    stack.enter_context(patch.object(train, "continuation_logits_batch", side_effect=logits))
                    stack.enter_context(patch.object(train, "target_prefix_states_batch",
                                                    return_value=torch.ones(batch_size, 7)))
                    kd = stack.enter_context(patch.object(train, "masked_answer_kd", wraps=train.masked_answer_kd))
                    ce = stack.enter_context(patch.object(train, "masked_gold_first_token_ce",
                                                         wraps=train.masked_gold_first_token_ce))
                    write = stack.enter_context(patch.object(train.base, "write_json_atomic"))
                    append = stack.enter_context(patch.object(train.base, "append_jsonl"))
                    stack.enter_context(patch.object(train.torch, "save"))
                    load = stack.enter_context(patch.object(train.torch, "load"))
                    stack.enter_context(patch.object(train.torch.cuda, "memory_reserved", return_value=0))
                    stack.enter_context(patch.object(train.torch.cuda, "max_memory_reserved", return_value=0))
                    stack.enter_context(patch.object(train, "tqdm", return_value=progress))
                    self.assertEqual(train.main(), 0)
                load.assert_not_called()
                full.assert_called_once_with(model, [part] * batch_size, False)
                row = append.call_args_list[0].args[1]
                direct = [i for i, reason in enumerate(row["reasoning"]) if not reason]
                reasoning = [i for i, reason in enumerate(row["reasoning"]) if reason]
                self.assertEqual(kd.call_args.args[2], reasoning)
                self.assertEqual(ce.call_args.args[2], direct)
                self.assertEqual(generated.call_count, int(bool(reasoning)))
                if reasoning:
                    self.assertEqual(generated.call_args.args[2], [teacher_pasts[i] for i in reasoning])
                self.assertEqual(row["supervision_counts"], {"gold_first_token_ce": len(direct),
                                                             "full_generated_kd": len(reasoning)})
                self.assertEqual(row["kd_batch_directions"], len(reasoning))
                self.assertEqual(row["gold_ce_batch_directions"], len(direct))
                for i, (prediction, trace) in enumerate(zip(captured["student"], captured["traces"])):
                    if i in direct:
                        expected = 1 if row["targets"][i] == "a" else 2
                        self.assertEqual(trace.tolist(), [expected, 6])
                        self.assertEqual(prediction.grad[1:].count_nonzero().item(), 0)
                        self.assertGreater(prediction.grad[0].abs().sum().item(), 0)
                    else:
                        self.assertEqual(trace.tolist(), [4, 5, 6])
                        self.assertTrue((prediction.grad.abs().sum(-1) > 0).all())
                for call in build.call_args_list:
                    self.assertIs(call.kwargs["enable_thinking"], False)
                config = write.call_args_list[0].args[1]
                self.assertEqual(config["supervision_protocol"],
                                 "no_reasoning_gold_first_token_ce__reasoning_full_generated_kd_v1")
                self.assertEqual(config["initialization_protocol"], "from_scratch")
                self.assertEqual(config["supervision_modes"]["no_reasoning"]["loss"], "gold_first_token_ce")
                self.assertEqual(config["supervision_modes"]["reasoning"]["max_new_tokens"], 64)
                for mode in config["supervision_modes"].values():
                    self.assertIs(mode["enable_thinking"], False)


if __name__=='__main__':unittest.main()