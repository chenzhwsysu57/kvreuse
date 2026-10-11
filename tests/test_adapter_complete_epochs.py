"""CPU-only tests of the actual training scheduler, without model imports."""
import ast
import math
import random
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scheduler():
    tree = ast.parse((ROOT / 'kv_semantic_attack/train_prefix_adapter_repeat.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'complete_epoch_batches')
    namespace = {'random': random}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-scheduler>', 'exec'), namespace)
    return namespace['complete_epoch_batches']


class CompleteEpochTests(unittest.TestCase):
    def check_pools(self, sizes):
        pools = {str(i): [(str(i), j) for j in range(size)] for i, size in enumerate(sizes)}
        expected = Counter(item for pool in pools.values() for item in pool)
        run = scheduler()
        for epoch in range(3):
            batches = run(pools, 4, 20261009 + epoch)
            self.assertEqual(len(batches), sum(math.ceil(size/4) for size in sizes))
            self.assertEqual(Counter(item for _, batch in batches for item in batch), expected)
            self.assertTrue(all(1 <= len(batch) <= 4 for _, batch in batches))
            self.assertTrue(all(item[0] == family for family, batch in batches for item in batch))
        self.assertEqual(run(pools, 4, 1), run(pools, 4, 1))

    def test_unequal_pools_and_partial_batches(self):
        self.check_pools([20, 7, 1])

    def test_five_scene_family_not_under_sampled(self):
        # 3 pairs x 2 roles x 2 modes per scene; five scenes share one family.
        self.check_pools([12] * 126 + [60, 60])

    def test_train75_complete_directions(self):
        self.check_pools([12] * 180)


if __name__ == '__main__':
    unittest.main()