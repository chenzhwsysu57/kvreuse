"""Both prompt variants must disable Qwen's native thinking mode."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from latent_remedy import eval_prefix_adapter_repeat as evaluation
from latent_remedy import train_prefix_adapter_repeat as training


class AdapterPromptModesTests(unittest.TestCase):
    def test_both_prompt_variants_disable_thinking(self):
        for module in (evaluation, training):
            for reasoning in (False, True):
                with self.subTest(module=module.__name__, reasoning=reasoning):
                    with patch.object(module.base, "build_prompt_parts") as build:
                        module.parts(None, {}, "a", reasoning)
                    self.assertIs(build.call_args.kwargs["enable_thinking"], False)
                    self.assertIs(build.call_args.kwargs["explicit_reasoning"], reasoning)
                    self.assertTrue(build.call_args.kwargs["boxed_output"])

    def test_brace_stop_only_for_non_json_answer_prefill(self):
        for reasoning in (False, True):
            self.assertEqual(evaluation.stop_at_closing_brace({}, reasoning), not reasoning)
            self.assertFalse(evaluation.stop_at_closing_brace({"output_protocol": "json_answer"}, reasoning))

    def test_explicit_reasoning_is_parsed_without_think_markers(self):
        text = r"Compute \text{Utility} and \frac{6}{2}. Final: \boxed{B}"
        tok = SimpleNamespace(decode=lambda *args, **kwargs: text)
        rec = {"dataset": "deal_or_no_deal", "gold_a": "B"}
        result = evaluation.generation_result(tok, rec, "a", [1], 4096, True)
        self.assertEqual(result["answer_text"], r"\boxed{B}")
        self.assertIn(r"\text{Utility}", result["reasoning_text"])
        self.assertTrue(result["correct"])
        self.assertFalse(result["hit_max_new_tokens"])

    def test_no_reasoning_restores_prefill_and_limit(self):
        tok = SimpleNamespace(decode=lambda *args, **kwargs: "B}")
        result = evaluation.generation_result(tok, {"dataset": "deal_or_no_deal", "gold_a": "B"}, "a", [1, 2], 2, False)
        self.assertEqual(result["output_text"], r"The answer is: \boxed{B}")
        self.assertTrue(result["correct"])
        self.assertTrue(result["hit_max_new_tokens"])

    def test_resume_refuses_legacy_outputs_without_modifying_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            samples = output / "samples.jsonl"
            samples.write_text('{"task_id":"old"}\n')
            with self.assertRaisesRegex(ValueError, "new output directory"):
                evaluation.ensure_protocol(output, {"enable_thinking": False})
            self.assertEqual(samples.read_text(), '{"task_id":"old"}\n')
            self.assertFalse((output / "eval_protocol.json").exists())

    def test_resume_requires_identical_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            protocol = {"enable_thinking": False, "explicit_reasoning": True}
            evaluation.ensure_protocol(output, protocol)
            evaluation.ensure_protocol(output, protocol)
            self.assertEqual(json.loads((output / "eval_protocol.json").read_text()), protocol)
            with self.assertRaisesRegex(ValueError, "new output directory"):
                evaluation.ensure_protocol(output, {**protocol, "explicit_reasoning": False})


if __name__ == "__main__":
    unittest.main()