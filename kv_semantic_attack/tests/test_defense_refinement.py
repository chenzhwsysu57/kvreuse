"""Offline tests for thresholded fixed-pool suffix selection."""

import unittest

from kv_semantic_attack.defense_refinement import choose_candidate, extract_scores, refinement_summary


def evaluation(direct, candidates):
    return {"direct_reuse": {"summary": {"overall": {"reuse_accuracy": direct}}},
            "candidates": [{"candidate": {"candidate_id": name, "suffix": suffix},
                            "summary": {"overall": {"reuse_accuracy": score}}}
                           for name, suffix, score in candidates]}


class RefinementTests(unittest.TestCase):
    def test_accepts_only_threshold_improvement(self):
        report = evaluation(0.10, [("near", "near", 0.129), ("winner", "winner", 0.14)])
        summary = refinement_summary(report, incumbent_suffix="", minimum_improvement_pp=3)
        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["selected"]["candidate_id"], "winner")
        self.assertAlmostEqual(summary["actual_improvement_pp"], 4.0)

    def test_retains_incumbent_on_small_or_worse_changes(self):
        report = evaluation(0.10, [("incumbent", "keep", 0.25), ("near", "new", 0.279)])
        summary = refinement_summary(report, incumbent_suffix="keep", minimum_improvement_pp=3)
        self.assertFalse(summary["accepted"])
        self.assertEqual(summary["selected"]["suffix"], "keep")
        with self.assertRaises(ValueError):
            choose_candidate(extract_scores(report), incumbent_suffix="missing", minimum_improvement_pp=1)


if __name__ == "__main__":
    unittest.main()