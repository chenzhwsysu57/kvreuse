"""Offline regressions for target anchoring and cumulative five-round feedback."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from kv_semantic_attack.adaptive_defense import SuffixDefender, defender_prompt, parse_candidates
from kv_semantic_attack.defense_refinement import candidate_feedback
from kv_semantic_attack.run_journal import file_ref
from kv_semantic_attack import run_defense_refinement as runner


def candidate(name="anchor", suffix="Follow the target rule before the shared data."):
    return {"candidate_id": name, "reasoning": "Observed leakage; test target anchoring.", "suffix": suffix}


def row(task, correct):
    return {"task_id": task, "target": "a", "attack_type": "test", "source_gold": "B",
            "full": {"correct": True, "prediction": "A", "gold": "A"},
            "reuse": {"correct": correct, "prediction": "A" if correct else "B"}}


def entry(corrects):
    return {"results": [row(str(i), value) for i, value in enumerate(corrects)],
            "summary": {"overall": {"reuse_accuracy": sum(corrects) / len(corrects)}}}


SOURCES = {str(i): {"task_id": str(i), "prefix_a": "target rule", "prefix_b": "source rule",
                   "shared_block": "data", "question": "choose"} for i in range(3)}


class DefenderImprovementTests(unittest.TestCase):
    def test_prompt_preserves_prefix_and_requests_evidence_and_history(self):
        text = defender_prompt({"refinement_history": [{"candidate": "old failed suffix"}]}, candidate_count=4)
        for phrase in ("target prefix 中的任务规则必须保留", "不包含完整任务", "不能真正清空",
                       "观测证据", "退化", "old failed suffix", "no-reasoning"):
            self.assertIn(phrase, text)

    def test_exact_count_and_unsafe_or_reserved_ids(self):
        class Fake:
            def chat_json(self, messages, **kwargs):
                self.kwargs = kwargs
                return {"candidates": [candidate()]}
        fake = Fake()
        with self.assertRaises(ValueError):
            SuffixDefender(fake).propose({}, candidate_count=4)
        self.assertEqual(fake.kwargs["max_tokens"], 3600)
        for name in ("../escape", "full", "direct_reuse", "incumbent"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_candidates({"candidates": [candidate(name)]})

    def test_invalid_suffix_retries_with_visible_text_requirement(self):
        class Fake:
            def __init__(self):
                self.calls = []
            def chat_json(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                return ({"candidates": [candidate("first", "\n\n")]}
                        if len(self.calls) == 1 else {"candidates": [candidate()]})
        fake = Fake()
        result = SuffixDefender(fake).propose({}, candidate_count=1)
        self.assertEqual(result[0].candidate_id, "anchor")
        self.assertEqual(fake.calls[0][1]["trace_name"], "adaptive_suffix_defender_attempt_1")
        self.assertIn("whitespace-only", fake.calls[1][0][0]["content"])

    def test_history_contains_repairs_regressions_and_rejected_candidate(self):
        report = {"direct_reuse": entry([True, False, False]), "candidates": [
            {"candidate": candidate(), **entry([False, True, False])}]}
        original = copy.deepcopy(report)
        feedback = candidate_feedback(report, SOURCES, incumbent_suffix="", examples_per_kind=1)
        self.assertEqual(report, original)
        changes = feedback[1]["comparisons"]["direct_reuse"]
        self.assertEqual(changes["counts"]["repair"], 1)
        self.assertEqual(changes["counts"]["regression"], 1)
        self.assertEqual(changes["counts"]["persistent_failure"], 1)
        self.assertEqual(changes["delta_pp"], 0)
        self.assertEqual(feedback[1]["candidate"]["suffix"], candidate()["suffix"])
        self.assertIn("direct_reuse:regression", feedback[1]["examples"])
        report["candidates"][0]["results"].pop()
        with self.assertRaises(ValueError):
            candidate_feedback(report, SOURCES, incumbent_suffix="")

    def run_mock_refinement(self, early_stop=False, parallel=False, target_accuracy=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jsonl"
            source.write_text("".join(json.dumps(r) + "\n" for r in SOURCES.values()))
            dashboard = root / "dashboard.json"
            dashboard.write_text('{"current_suffix":""}')
            run_dir = root / "run"
            seen = []
            originals = []

            def fake_run(command):
                def arg(flag):
                    return Path(command[command.index(flag) + 1])
                if "run_adaptive_defender.py" in command[1]:
                    seen.append(json.loads(arg("--dashboard").read_text()))
                    payload = {"candidates": [candidate(f"new_{len(seen)}", f"new suffix {len(seen)}")]}
                    arg("--output").write_text(json.dumps(payload))
                    originals.append(file_ref(arg("--output")))
                else:
                    if parallel:
                        self.assertIn("run_multi_gpu_defense.py", command[1])
                        self.assertEqual(command[command.index("--gpu-slots") + 1:], ["1=2", "2=2", "3=2"])
                        self.assertIn("--output-dir", command)
                    candidates = json.loads(arg("--candidates").read_text())["candidates"]
                    # Round 1 improves by 33pp, later proposals regress, so the
                    # incumbent must be retained and all failures still fed back.
                    report = {"model": "mock", "direct_reuse": entry([False, False, False]),
                              "candidates": [{"candidate": c, **entry([c["suffix"] == "new suffix 1", False, False])}
                                             for c in candidates]}
                    arg("--output").write_text(json.dumps(report))

            argv = ["refine", "--input", str(source), "--dashboard", str(dashboard),
                    "--run-dir", str(run_dir), "--start-step", "1", "--candidates-per-attempt", "1"]
            if early_stop:
                argv.append("--stop-on-threshold")
            if target_accuracy is not None:
                argv.extend(["--target-accuracy", str(target_accuracy)])
            if parallel:
                argv.extend(["--gpu-slots", "1=2", "2=2", "3=2"])
            with patch.object(sys, "argv", argv), patch.object(runner, "run", fake_run), patch("builtins.print"):
                self.assertEqual(runner.main(), 0)
            result = json.loads((run_dir / "refinement_summary.json").read_text())
            self.assertEqual(result["rounds_completed"], 1 if early_stop or target_accuracy is not None else 5)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["suffix"], "new suffix 1")
            for ref in originals:
                self.assertEqual(file_ref(Path(ref["path"])), ref)
            if not early_stop and target_accuracy is None:
                self.assertEqual(len(seen[-1]["refinement_history"]), 4)
                self.assertIn("new suffix 2", json.dumps(seen[-1]["refinement_history"]))
                self.assertEqual(seen[-1]["current_suffix"], "new suffix 1")

    def test_default_five_rounds_and_immutable_candidates(self):
        self.run_mock_refinement()

    def test_explicit_early_stop(self):
        self.run_mock_refinement(early_stop=True)

    def test_five_rounds_parallel_gpu_slots(self):
        self.run_mock_refinement(parallel=True)

    def test_absolute_target_stops_without_threshold(self):
        self.run_mock_refinement(target_accuracy=0.3)


if __name__ == "__main__":
    unittest.main()