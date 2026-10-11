"""CPU tensor tests for the unchanged vendored row-local cache implementation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY_ERROR = None
try:
    import torch
    from transformers import DynamicCache
except ImportError as error:
    DEPENDENCY_ERROR = str(error)
else:
    if not hasattr(DynamicCache(), "key_cache"):
        DEPENDENCY_ERROR = "Requires legacy DynamicCache.key_cache (requirements-relay.txt: transformers 4.52.4)"
    else:
        # Import the actual utility module without the package's LLM/model stack.
        spec = importlib.util.spec_from_file_location(
            "batch_cache_under_test", ROOT / "third_party/RelayCaching/RelayCaching/utils/relay_utils.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        BatchedRelayCache = module.BatchedRelayCache


@unittest.skipIf(DEPENDENCY_ERROR is not None, DEPENDENCY_ERROR or "")
class RelayBatchCacheTests(unittest.TestCase):
    def cache(self, mode="epic"):
        prefix = DynamicCache()
        values = torch.arange(24, dtype=torch.float32).reshape(2, 1, 6, 2)
        for layer in range(3):
            prefix.update(values.clone(), values.clone(), layer)
        indices = torch.tensor([[3, 4, 0], [1, 2, 3]])
        valid = torch.tensor([[True, True, False], [True, True, True]])
        # CPU-only even on GPU hosts: avoid legacy RelayCache's optional CUDA stream.
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            return BatchedRelayCache(
                prefix, row_lengths=torch.tensor([4, 6]),
                recompute_indices=indices, recompute_valid=valid,
                relay_absolute_indices=indices, relay_valid=valid,
                blank_absolute_indices=torch.empty((2, 0), dtype=torch.long),
                blank_valid=torch.empty((2, 0), dtype=torch.bool),
                append_physical_indices=torch.tensor([[5], [5]]),
                append_valid=torch.ones((2, 1), dtype=torch.bool),
                execution_mode=mode, max_length=6, cacheblend_diff_por=0.5,
                det_layer=0, tau_dev=1.0)

    def test_rowwise_gather_scatter_and_padded_updates(self):
        cache = self.cache()
        before = cache.key_cache[0].clone()
        updates = torch.tensor([[[[101., 102.], [103., 104.], [999., 999.]]],
                                [[[201., 202.], [203., 204.], [205., 206.]]]])
        cache.update_cache(updates, updates + 10, 0)
        expected = before.clone()
        expected[0, :, 3:5] = updates[0, :, :2]
        expected[1, :, 1:4] = updates[1]
        torch.testing.assert_close(cache.key_cache[0], expected)
        self.assertTrue(torch.equal(cache.key_cache[0][0, :, 0], before[0, :, 0]))
        gathered = cache._gather_rows(cache.key_cache[0], cache.recompute_indices_by_row)
        torch.testing.assert_close(gathered[0, :, :2], updates[0, :, :2])
        torch.testing.assert_close(gathered[1], updates[1])
        self.assertEqual(cache.recompute_num.tolist(), [2, 3])
        self.assertEqual(cache.row_lengths.tolist(), [4, 6])

    def test_unequal_length_decode_append_is_rejected(self):
        cache = self.cache()
        cache.recompute_label = False
        with self.assertRaisesRegex(RuntimeError, "compacted equal-length"):
            cache.update_cache(torch.ones(2, 1, 1, 2), torch.ones(2, 1, 1, 2), 0)

    def test_relaycaching_deviation_selection_is_row_local(self):
        cache = self.cache("relaycaching")
        old = cache._gather_rows(cache.value_cache[0], cache.relay_absolute_indices_by_row)
        correction = old.clone()
        correction[0, :, 1] *= -1
        correction[1, :, 0] *= -1
        query = torch.cat([correction, torch.ones(2, 1, 1, 2)], dim=2)
        cache.update_cache(query, query, 0)
        self.assertEqual(cache._batch_active_indices[:, 0].tolist(), [4, 1])
        self.assertEqual(cache._batch_active_valid.sum(dim=1).tolist(), [1, 1])
        self.assertEqual(cache.recompute_num.tolist(), [1, 1])

    def test_cacheblend_selection_and_causal_mask_are_row_local(self):
        cache = self.cache("cacheblend")
        old = cache._gather_rows(cache.value_cache[1], cache.relay_absolute_indices_by_row)
        # Different winning token in each row; padded score must not participate.
        correction = old + torch.tensor([[[[1., 0.], [9., 0.], [999., 999.]]],
                                         [[[1., 0.], [2., 0.], [8., 0.]]]])
        query = torch.cat([correction, torch.ones(2, 1, 1, 2)], dim=2)
        cache.update_cache(query, query, 1)
        self.assertEqual(cache._batch_active_indices.tolist(), [[4], [3]])
        self.assertEqual(cache.recompute_num.tolist(), [1, 1])
        hidden = torch.ones(2, 4, 2)
        hidden, mask, _, positions = cache.pre_drop_process(
            hidden, torch.zeros(2, 1, 4, 6), None, None, lambda h, p: (p, p))
        self.assertEqual(cache.query_valid_by_row.sum(dim=1).tolist(), [1, 1])
        self.assertEqual(positions[:, 0].tolist(), [2, 3])
        self.assertEqual((mask[:, 0, 0] == 0).tolist(),
                         [[False, False, True, True, True, False],
                          [True, True, True, True, False, False]])
        self.assertTrue((mask[:, :, 1:3] < 0).all())
        self.assertTrue((hidden[:, 1:3] == 0).all())
        # Deep-layer update corrects only the selected relay token in each row.
        before = cache.key_cache[2].clone()
        cache.update_cache(hidden[:, None], hidden[:, None], 2)
        for row, selected in enumerate((4, 3)):
            expected = before[row].clone()
            expected[:, selected] = 1
            expected[:, 5] = 1  # final prompt append, not a dynamic decode token
            torch.testing.assert_close(cache.key_cache[2][row], expected)


if __name__ == "__main__":
    unittest.main()