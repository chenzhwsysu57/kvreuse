"""Offline prompt search tests: real Optuna, fake inference, no model/API/GPU."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import optuna

from kv_semantic_attack.eval_synthetic_batch import raw_batch_adapter
from kv_semantic_attack.prompt_search import (
    BASELINES, INTRODUCTIONS, PrefixCompressor, PromptConfig, optimize_prompts,
    prefix_key, render_prompt, search_space, select_shortest, split_records,
)
from kv_semantic_attack.run_prompt_search import CachedEvaluator, check_rows, main, run_experiment
from kv_semantic_attack.synthetic_tasks import TASK_TYPES, generate_tasks


class FakeBackend:
    def __init__(self):
        self.full_calls = 0
        self.reuse_calls = 0

    def token_count(self, text):
        return len(text.split())

    def full(self, records):
        self.full_calls += 1
        return [self.row(record, side, "full", True) for record in records for side in ("a", "b")], 0.0

    def reuse(self, records, builder, label):
        self.reuse_calls += 1
        rows = []
        for record in records:
            for side in ("a", "b"):
                bridge = builder(record[f"prefix_{side}"])
                rows.append(self.row(record, side, "reuse", bool(bridge)))
        return rows, {"wall_seconds": 0.0}

    @staticmethod
    def row(record, side, method, correct):
        return {"task_id": record["task_id"], "target": side, "attack_type": record["attack_type"],
                "source_gold": record["gold_b" if side == "a" else "gold_a"],
                method: {"correct": correct, "prediction": record[f"gold_{side}"] if correct else "Z",
                         "hit_max_new_tokens": False}}


class PrefixPromptTests(unittest.TestCase):
    def test_conservative_compression_preserves_conditions_and_literals(self):
        compressor = PrefixCompressor()
        body = 'Use only EU rows; maximize score, break ties by minimum cost. Output "A  B" if false.'
        for opening in INTRODUCTIONS:
            self.assertEqual(compressor.compress(opening + body, "compact"), (body, False))
        self.assertEqual(compressor.compress(body, "compact"), (body, True))
        self.assertEqual(compressor.compress(body, "full"), (body, False))

    def test_bank_uses_exact_prefix_hash_and_falls_back(self):
        prefix = "a task"
        compressor = PrefixCompressor({"short": {prefix_key(prefix): "task"}})
        self.assertEqual(compressor.compress(prefix, "short"), ("task", False))
        self.assertEqual(compressor.compress(prefix + " ", "short"), (prefix + " ", True))
        with self.assertRaises(ValueError):
            PrefixCompressor({"full": {}})
        with self.assertRaises(ValueError):
            PrefixCompressor({"short": {prefix_key(prefix): ""}})

    def test_adapter_target_only_no_mutation_full_unchanged_and_restored(self):
        def original(*args):
            return None
        base = SimpleNamespace(result_from_generation=original,
                               build_prompt_parts=lambda tokenizer, record, side, **kwargs: record)
        batch = SimpleNamespace(base=base, build_parts=original)
        record = {"prefix_a": INTRODUCTIONS[0] + "maximize score",
                  "prefix_b": INTRODUCTIONS[1] + "minimize cost", "shared_block": "unchanged",
                  "question": "Choose a letter", "metadata": {"secret": "not passed"}, "gold_a": "A"}
        before = copy.deepcopy(record)
        seen = []

        def builder(prefix):
            self.assertIsInstance(prefix, str)
            seen.append(prefix)
            return render_prompt(prefix, PromptConfig(), PrefixCompressor())

        with raw_batch_adapter(batch, prefix_prompt_builder=builder):
            for side, objective in (("a", "maximize score"), ("b", "minimize cost")):
                changed = batch.build_parts(None, record, side, "reuse")
                self.assertTrue(changed["question"].startswith("Current task:\n" + objective))
                self.assertTrue(changed["question"].endswith("\n\nChoose a letter"))
                for key in ("prefix_a", "prefix_b", "shared_block"):
                    self.assertEqual(changed[key], record[key])
            self.assertEqual(batch.build_parts(None, record, "a", "full"), before)
        self.assertEqual(seen, [record["prefix_a"], record["prefix_b"]])
        self.assertEqual(record, before)
        self.assertIs(batch.build_parts, original)
        with self.assertRaises(ValueError):
            with raw_batch_adapter(batch, suffix="legacy", prefix_prompt_builder=builder):
                pass

    def test_search_requires_both_restatement_and_generic(self):
        space = search_space(PrefixCompressor())
        self.assertEqual(len(space), 24)
        self.assertTrue(all(item.compression != "none" and item.generic != "none" for item in space))
        self.assertEqual(render_prompt("P", BASELINES["direct_reuse"], PrefixCompressor()), "")


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cls.records = list(generate_tasks({name: 5 for name in TASK_TYPES}, seed=51))

    def test_group_split_deterministic_and_leak_free(self):
        records = copy.deepcopy(self.records)
        extra = copy.deepcopy(records[0])
        extra["task_id"] += "-another-layout"
        records.append(extra)
        splits = split_records(records, seed=42)
        self.assertEqual(splits, split_records(list(reversed(records)), seed=42))
        group_sets = [{row["shared_data_id"] for row in rows} for rows in splits.values()]
        self.assertFalse(group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
        for rows in splits.values():
            self.assertEqual({row["attack_type"] for row in rows}, set(TASK_TYPES))
        with self.assertRaises(ValueError):
            split_records(records + [records[0]], seed=42)
        with self.assertRaises(ValueError):
            split_records(records[:1], seed=42)
        with self.assertRaises(ValueError):
            split_records(records, seed=42, validation_fraction=.6, test_fraction=.5)

    def test_validation_tolerance_selects_shortest(self):
        values = [{"accuracy": a, "mean_bridge_tokens": n, "config": {"id": i}}
                  for i, a, n in ((0, .9, 100), (1, .895, 20), (2, .8, 1))]
        self.assertEqual(select_shortest(values, 0)["config"]["id"], 0)
        self.assertEqual(select_shortest(values, 1)["config"]["id"], 1)
        with self.assertRaises(ValueError):
            select_shortest(values, float("nan"))

    def test_tpe_and_random_resume_deterministically(self):
        def evaluate(config):
            return {"accuracy": .7 + .1 * (config.compression == "compact"), "mean_bridge_tokens": 10}

        for sampler in ("tpe", "random"):
            whole = optuna.create_study(direction="maximize")
            expected = optimize_prompts(whole, PrefixCompressor(), evaluate, budget=12, seed=72, sampler=sampler)
            with tempfile.TemporaryDirectory() as tmp:
                storage = "sqlite:///" + str(Path(tmp) / "study.db")
                study = optuna.create_study(storage=storage, study_name="test", direction="maximize")
                optimize_prompts(study, PrefixCompressor(), evaluate, budget=5, seed=72, sampler=sampler)
                resumed = optuna.load_study(storage=storage, study_name="test")
                actual = optimize_prompts(resumed, PrefixCompressor(), evaluate, budget=12, seed=72, sampler=sampler)
                self.assertEqual([item["config"] for item in expected], [item["config"] for item in actual])
                self.assertEqual(len({json.dumps(item["config"], sort_keys=True) for item in actual}), 12)

    def test_failed_trial_can_resume(self):
        study = optuna.create_study(direction="maximize")

        def fail(config):
            raise RuntimeError("interrupted evaluation")

        with self.assertRaises(RuntimeError):
            optimize_prompts(study, PrefixCompressor(), fail, budget=2, seed=1, sampler="tpe")
        values = optimize_prompts(study, PrefixCompressor(), lambda c: {"accuracy": .5, "mean_bridge_tokens": 2},
                                  budget=2, seed=1, sampler="tpe")
        self.assertEqual(len(values), 2)

    def test_direction_validation_rejects_missing_and_duplicate(self):
        rows, _ = FakeBackend().full(self.records[:1])
        check_rows(rows, self.records[:1], "full")
        for broken in (rows[:1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                check_rows(broken, self.records[:1], "full")

    def test_identical_rendered_text_uses_cache_and_single_full(self):
        records = [dict(self.records[0], prefix_a="a task", prefix_b="b task")]
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeBackend()
            evaluator = CachedEvaluator(Path(tmp), {"search": records}, PrefixCompressor(), backend, "contract")
            full = evaluator.evaluate("search", PromptConfig("full"))
            compact = evaluator.evaluate("search", PromptConfig("compact"))
            self.assertEqual(full["evaluation_file"], compact["evaluation_file"])
            self.assertEqual(backend.full_calls, 1)
            self.assertEqual(backend.reuse_calls, 1)
            self.assertEqual(compact["prompt_artifacts"]["fallback_directions"], 2)

    def test_end_to_end_freezes_selection_before_test_and_resumes(self):
        splits = split_records(self.records, seed=72)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)

            class CheckedBackend(FakeBackend):
                def full(self, records):
                    if records == splits["test"]:
                        self.assert_selection_exists()
                    return super().full(records)

                def assert_selection_exists(self):
                    if not (output / "selection.json").exists():
                        raise AssertionError("test was read before selection froze")

            backend = CheckedBackend()
            kwargs = dict(contract="test", budget=4, seed=72, sampler="tpe", shortlist_size=2, tolerance_pp=0)
            result = run_experiment(output, splits, PrefixCompressor(), backend, **kwargs)
            self.assertTrue(result["complete"])
            self.assertEqual(backend.full_calls, 3)
            self.assertEqual(result["test"]["accuracy"], 1)
            counts = backend.full_calls, backend.reuse_calls
            self.assertEqual(run_experiment(output, splits, PrefixCompressor(), backend, **kwargs), result)
            self.assertEqual((backend.full_calls, backend.reuse_calls), counts)
            # Simulate interruption just before writing final summary: cached
            # test results and frozen selection must still be reused unchanged.
            (output / "summary.json").unlink()
            self.assertEqual(run_experiment(output, splits, PrefixCompressor(), backend, **kwargs), result)
            self.assertEqual((backend.full_calls, backend.reuse_calls), counts)

    def test_dry_run_no_model_or_output_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "tasks.jsonl"
            input_path.write_text("\n".join(json.dumps(row) for row in self.records), encoding="utf-8")
            output = Path(tmp) / "not-created"
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                code = main(["--input", str(input_path), "--output-dir", str(output), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            self.assertTrue(json.loads(stream.getvalue())["dry_run"])


if __name__ == "__main__":
    unittest.main()