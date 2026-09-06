"""Offline tests of raw evaluation and temporary batch adapters."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from kv_semantic_attack.eval_synthetic_batch import (
    evaluate_chunk, parse_args, raw_batch_adapter, raw_result, summarize,
    without_shared_data_tags,
)


class RawEvaluationTests(unittest.TestCase):
    def test_remove_only_boundary_tags(self):
        record = {"shared_block": "<shared_data>\nR1:42\n</shared_data>",
                  "prefix_a": "minimum", "gold_a": "R1"}
        changed = without_shared_data_tags(record)
        self.assertEqual(changed["shared_block"], "\nR1:42\n")
        self.assertEqual(changed["prefix_a"], record["prefix_a"])
        self.assertEqual(changed["gold_a"], record["gold_a"])
        self.assertTrue(record["shared_block"].startswith("<shared_data>"))
        self.assertEqual(without_shared_data_tags(changed), changed)
        with self.assertRaises(ValueError):
            without_shared_data_tags({"shared_block": "<shared_data>partial"})

    def test_eos_trim_and_preserve_case(self):
        tokenizer = SimpleNamespace(eos_token_id=9, decode=lambda ids, **kwargs: "AbC}")
        result = raw_result(tokenizer, "x", "AbC", [1, 9, 9, 9], True, 0, 4, False,
                            output_prefix="The answer is: \\boxed{")
        self.assertEqual(result["output_text"], "The answer is: \\boxed{AbC}")
        self.assertEqual(result["prediction"], "AbC")
        self.assertEqual(result["output_token_ids"], [1])
        self.assertFalse(result["hit_max_new_tokens"])
        self.assertTrue(result["correct"])
        self.assertFalse(raw_result(tokenizer, "x", "abc", [1], True, 0, 1, False,
                                    output_prefix="The answer is: \\boxed{")["correct"])
        self.assertTrue(raw_result(tokenizer, "x", "AbC", [1], True, 0, 1, False,
                                   output_prefix="The answer is: \\boxed{")["hit_max_new_tokens"])

    def test_boxed_extraction_supports_json_answer(self):
        tokenizer = SimpleNamespace(eos_token_id=9, decode=lambda ids, **kwargs: '{"id":"R1","value":4}}')
        result = raw_result(tokenizer, "x", '{"id":"R1","value":4}', [1, 9], True, 0, 2, False,
                            output_prefix="The answer is: \\boxed{")
        self.assertEqual(result["prediction"], '{"id":"R1","value":4}')
        self.assertTrue(result["correct"])

    def test_unclosed_single_letter_matches_batch_eval_protocol(self):
        tokenizer = SimpleNamespace(eos_token_id=9, decode=lambda ids, **kwargs: "C")
        result = raw_result(tokenizer, "x", "C", [1], True, 0, 2, False,
                            output_prefix="The answer is: \\boxed{")
        self.assertEqual(result["prediction"], "C")
        self.assertTrue(result["correct"])

    def test_adapter_restores_and_forbids_repair(self):
        original_build = lambda *args: None
        original_result = lambda *args: None
        base = SimpleNamespace(result_from_generation=original_result,
                               build_prompt_parts=lambda *args, **kwargs: kwargs)
        batch = SimpleNamespace(base=base, build_parts=original_build)
        with raw_batch_adapter(batch):
            self.assertIs(batch.base.result_from_generation, raw_result)
            options = batch.build_parts(None, {}, "a", "reuse")
            self.assertTrue(options["boxed_output"])
            self.assertFalse(options["enable_thinking"])
            with self.assertRaises(ValueError):
                batch.build_parts(None, {}, "a", "ours_post")
        self.assertIs(batch.build_parts, original_build)
        self.assertIs(batch.base.result_from_generation, original_result)

    def test_summary(self):
        rows = [
            {"attack_type": "field_switch", "source_gold": "B",
             "full": {"correct": True, "hit_max_new_tokens": False},
             "reuse": {"correct": False, "prediction": "B", "hit_max_new_tokens": False}},
            {"attack_type": "field_switch", "source_gold": "A",
             "full": {"correct": False, "hit_max_new_tokens": True},
             "reuse": {"correct": True, "prediction": "B", "hit_max_new_tokens": False}},
        ]
        summary = summarize(rows)["overall"]
        self.assertEqual(summary["full_accuracy"], 0.5)
        self.assertEqual(summary["reuse_accuracy"], 0.5)
        self.assertEqual(summary["reuse_failure_rate_on_full_correct"], 1)
        self.assertEqual(summary["source_gold_match_rate"], 0.5)
        self.assertEqual(summarize([]), {})

    def test_single_method_summary_does_not_invent_comparison(self):
        for method in ("full", "reuse"):
            row = {"attack_type": "field_switch", "source_gold": "B",
                   method: {"correct": True, "prediction": "A", "hit_max_new_tokens": False}}
            summary = summarize([row])["overall"]
            other = "reuse" if method == "full" else "full"
            self.assertEqual(summary[f"{method}_accuracy"], 1.0)
            self.assertIsNone(summary[f"{other}_accuracy"])
            self.assertIsNone(summary["full_minus_reuse_pp"])
            self.assertIsNone(summary["reuse_failures_on_full_correct"])

    def test_fast_defaults_and_explicit_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / "report.json")
            args = parse_args(["--output", output])
            self.assertEqual(args.batch_size, 16)
            self.assertEqual(args.verify_per_type, 0)
            self.assertEqual(args.save_every, 0)
            for method in ("full", "reuse"):
                args = parse_args(["--output", output, "--method", method])
                self.assertEqual(args.method, method)
            with self.assertRaises(SystemExit):
                parse_args(["--output", output, "--method", "reuse", "--verify-per-type", "1"])
            with self.assertRaises(SystemExit):
                parse_args(["--output", output, "--save-every", "-1"])

    def test_only_requested_method_runs(self):
        for method in ("full", "reuse"):
            batch = SimpleNamespace(
                torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock())),
                full_rows=Mock(return_value=(["full_a", "full_b"], 0)),
                reuse_rows=Mock(return_value=["reuse_a", "reuse_b"]),
            )
            timing = {method: 0.0}
            values = evaluate_chunk(batch, None, None, [{"task_id": "x"}], [method], 128, timing, "reference")
            self.assertEqual(list(values), [method])
            if method == "full":
                batch.full_rows.assert_called_once()
                batch.reuse_rows.assert_not_called()
            else:
                batch.reuse_rows.assert_called_once()
                batch.full_rows.assert_not_called()


if __name__ == "__main__":
    unittest.main()