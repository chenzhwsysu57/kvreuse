"""Gap-first diagnostic feedback; no API or GPU required."""
import unittest

from kv_semantic_attack.build_adaptive_dashboard import build_dashboard, diverse_examples
from kv_semantic_attack.defense_refinement import candidate_feedback


def make_row(name, index, full, reuse, side="a"):
    return {"task_id": f"{name}-{index}", "attack_type": name, "target": side, "source_gold": "B",
            "full": {"correct": full, "gold": "A", "prediction": "A" if full else "C"},
            "reuse": {"correct": reuse, "prediction": "A" if reuse else "B"}}


def sources_for(rows):
    return {r["task_id"]: {"prefix_a": "target", "prefix_b": "source", "shared_block": "data",
                           "question": "choose", "metadata": {"layout": "table"}} for r in rows}


class GapPriorityTests(unittest.TestCase):
    def test_gap_then_sample_size_not_alphabetical(self):
        rows = [make_row("a_low", 0, True, True), make_row("a_low", 1, True, False),
                make_row("z_high", 0, True, False), make_row("z_high", 1, True, False),
                make_row("b_small", 0, True, False), make_row("negative", 0, False, True)]
        report = {"model": "mock", "direct_reuse": {"results": rows}, "candidates": []}
        sources = sources_for(rows)
        d = build_dashboard(report, sources, candidate_id="direct_reuse", examples_per_type=6)
        expected = ["z_high", "b_small", "a_low", "negative"]
        self.assertEqual(list(d["by_type"]), expected)
        self.assertEqual(list(d["failure_examples"]), expected)
        self.assertEqual([v["attack_type"] for v in d["failure_priority"]], expected)
        self.assertEqual(d["by_type"]["a_low"]["full_minus_reuse_pp"], 50)
        self.assertEqual(d["failure_examples"]["negative"], [])
        zero = build_dashboard(report, sources, candidate_id="direct_reuse", examples_per_type=0)
        self.assertTrue(all(not values for values in zero["failure_examples"].values()))

    def test_rerank_using_selected_suffix(self):
        rows = [make_row("a", 0, True, False), make_row("z", 0, True, True)]
        changed = [make_row("a", 0, True, True), make_row("z", 0, True, False)]
        report = {"model": "mock", "direct_reuse": {"results": rows}, "candidates": [
            {"candidate": {"candidate_id": "new", "suffix": "test"}, "results": changed}]}
        dashboard = build_dashboard(report, sources_for(rows), candidate_id="new", examples_per_type=6)
        self.assertEqual(dashboard["failure_priority"][0]["attack_type"], "z")
        self.assertEqual(dashboard["failure_examples"]["a"], [])

    def test_direction_balance_dedup_and_input_order_independence(self):
        rows = [make_row("x", i, True, False, side) for side in ("a", "b") for i in range(8)]
        sources = sources_for(rows)
        sources["x-1"]["metadata"]["layout"] = "json"
        sources["x-2"]["shared_block"] = "long" * 500
        rows[3]["reuse"]["prediction"] = "C"
        chosen = diverse_examples(rows, sources, 6)
        self.assertEqual(sum(r["target"] == "a" for r in chosen), 3)
        self.assertEqual(sum(r["target"] == "b" for r in chosen), 3)
        self.assertEqual(chosen, diverse_examples(list(reversed(rows)), sources, 6))
        self.assertEqual(chosen, diverse_examples(rows + rows, sources, 6))
        self.assertIn("json", [sources[r["task_id"]]["metadata"]["layout"] for r in chosen])
        one_side = [r for r in rows if r["target"] == "a"]
        self.assertEqual(len(diverse_examples(one_side, sources, 6)), 6)

    def test_history_examples_prioritize_gap_without_losing_regressions(self):
        rows = [make_row("a_low", 0, True, False), make_row("a_low", 1, True, True),
                make_row("z_high", 0, True, False), make_row("z_high", 1, True, False)]
        baseline = [make_row("a_low", 0, True, True), *rows[1:]]
        report = {"direct_reuse": {"results": baseline, "summary": {}}, "candidates": [
            {"candidate": {"candidate_id": "new", "suffix": "test"}, "results": rows, "summary": {}}]}
        feedback = candidate_feedback(report, sources_for(rows), incumbent_suffix="", examples_per_kind=1)[1]
        self.assertEqual(feedback["failure_priority"][0]["attack_type"], "z_high")
        examples = feedback["examples"]
        self.assertEqual(examples["direct_reuse:persistent_failure"][0]["attack_type"], "z_high")
        self.assertEqual(examples["direct_reuse:regression"][0]["attack_type"], "a_low")


if __name__ == "__main__":
    unittest.main()