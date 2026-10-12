"""CPU-only compact-input compatibility and result/resume regression tests."""

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run
from kvreuse_data.schema import CORE_FIELDS, normalize_record, validate_record
from latent_remedy import eval_prefix_adapter_repeat as adapter_eval
from latent_remedy import eval_universal_soft_suffix as suffix_eval
from latent_remedy import train_prefix_adapter_repeat as adapter_train
from latent_remedy import train_universal_soft_suffix as suffix_train
from scripts import batch_eval, relay_deal_adapter, run_direct_reuse, validate_datasets


ROOT = Path(__file__).resolve().parents[1]


def compact(**changes):
    return {
        "dataset": "family_001", "prefix_a": "角色 A", "prefix_b": "角色 B",
        "shared_block": "[A] first\n[B] second", "question": "Choose one.",
        "gold_a": "A", "gold_b": "B", **changes,
    }


class CompactDataTests(unittest.TestCase):
    def test_copy_canonical_digest_and_idempotence(self):
        record = compact()
        before = copy.deepcopy(record)
        normalized = normalize_record(record)
        payload = {key: record[key] for key in CORE_FIELDS}
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"))
        expected = "sample-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.assertEqual(normalized["task_id"], expected)
        self.assertEqual(normalized["metric"], "exact_match")
        self.assertEqual(record, before)
        self.assertIsNot(normalized, record)
        self.assertEqual(normalize_record(normalized), normalized)

    def test_cross_file_ids_ignore_path_order_and_noncore_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            ids = []
            for index, name in enumerate(("full.jsonl", "train.jsonl", "test.jsonl", "OODcombined.jsonl")):
                record = compact()
                if index % 2:
                    record = dict(reversed(list(record.items())))
                record.update(metadata={"source": name}, case_id=name, attack_type=name)
                path = Path(temporary) / name
                path.write_text(json.dumps(record, ensure_ascii=index % 2 == 0, indent=None) + "\n")
                ids.append(run.load_jsonl(path)[0]["task_id"])
            self.assertEqual(len(set(ids)), 1)

    def test_every_core_text_and_protocol_affects_id(self):
        original = compact()
        reference = normalize_record(original)["task_id"]
        for key in CORE_FIELDS:
            with self.subTest(field=key):
                changed = {**original, key: original[key] + " changed"}
                self.assertNotEqual(normalize_record(changed)["task_id"], reference)
        protocols = (None, "option_letter", "raw_payload", "final_payload", "json_answer", "bracket_payload")
        ids = [normalize_record(original if protocol is None else
                                {**original, "output_protocol": protocol})["task_id"]
               for protocol in protocols]
        self.assertEqual(len(set(ids)), len(protocols))

    def test_explicit_id_and_legacy_fields_preserved(self):
        record = compact(task_id="legacy-id", metric="exact_match", metadata={"source": 1})
        self.assertEqual(normalize_record(record), record)
        self.assertIsNot(normalize_record(record), record)
        self.assertEqual(normalize_record(compact(task_id="legacy-id"))["task_id"], "legacy-id")

    def test_validate_compact_and_reject_malformed_without_mutation(self):
        record = compact()
        validate_record(record)
        self.assertNotIn("task_id", record)
        malformed = [None, [], "text", 3]
        for key in CORE_FIELDS:
            missing = compact()
            del missing[key]
            malformed.append(missing)
        for key in (*CORE_FIELDS, "task_id", "metric"):
            for value in (None, "", "  ", 1, [], {}):
                malformed.append(compact(**{key: value}))
        malformed.extend([compact(gold_b="A"), compact(metric="accuracy")])
        malformed.extend(compact(output_protocol=value)
                         for value in (None, "", "unknown", [], {}, 1))
        for record in malformed:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    validate_record(record)
                with self.assertRaises(ValueError):
                    normalize_record(record)

    def test_all_input_readers_and_json_list_support_compact(self):
        readers = (run.load_jsonl, run_direct_reuse.load_jsonl, relay_deal_adapter.load_jsonl,
                   suffix_train.read_records, adapter_train.read_records, adapter_eval.read_records,
                   suffix_eval.read_records, batch_eval.base.load_jsonl)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.jsonl"
            records = [compact(dataset=dataset, output_protocol="option_letter")
                       for dataset in ("argkp", "deal_or_no_deal", "job_interview", "family_240")]
            path.write_text("\n" + "\n\n".join(map(json.dumps, records)) + "\n")
            expected = [normalize_record(record) for record in records]
            for reader in readers:
                with self.subTest(reader=reader):
                    self.assertEqual(reader(path), expected)
            json_path = path.with_suffix(".json")
            json_path.write_text(json.dumps(records))
            self.assertEqual(suffix_train.read_records(json_path), expected)
            self.assertEqual(len(list(relay_deal_adapter.iter_examples(expected))), 8)
            self.assertEqual(len(suffix_eval.rows(expected)), 8)
            path.write_text(json.dumps(compact(gold_a=3)) + "\n")
            for reader in readers:
                with self.subTest(malformed_reader=reader), self.assertRaises(ValueError):
                    reader(path)
            for value in ({}, [None]):
                json_path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    suffix_train.read_records(json_path)

    def test_synthetic_240_families_do_not_collapse(self):
        records = [normalize_record(compact(dataset=f"family_{index:03d}"))
                   for index in range(1, 241)]
        families = {suffix_train.task_family(record) for record in records}
        self.assertEqual(len(families), 240)
        pools = suffix_train.balanced_row_pools(suffix_train.directions(records), seed=7)
        self.assertEqual(len(pools), 240)
        self.assertTrue(all(len(pool) == 2 for pool in pools.values()))
        self.assertEqual(suffix_train.task_family(compact(attack_type="old-rule")),
                         "synthetic:old-rule")
        self.assertEqual(suffix_train.task_family(compact(metadata={"attack_type": "old-rule"})),
                         "synthetic:old-rule")

    def test_validator_skips_absent_source_metadata_but_keeps_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "compact.jsonl"
            record = compact(dataset="job_interview")
            path.write_text(json.dumps(record) + "\n")
            with patch.object(sys, "argv", ["validate_datasets.py", str(path)]), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(validate_datasets, "validate_job_interview_record") as oracle:
                self.assertEqual(validate_datasets.main(), 0)
                oracle.assert_not_called()
            for bad in (compact(dataset="job_interview", gold_a=None),
                        compact(dataset="job_interview", metadata={})):
                path.write_text(json.dumps(bad) + "\n")
                with patch.object(sys, "argv", ["validate_datasets.py", str(path)]), \
                        self.assertRaises(ValueError):
                    validate_datasets.main()
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
            with patch.object(sys, "argv", ["validate_datasets.py", str(path)]), \
                    self.assertRaisesRegex(ValueError, "duplicate task_id"):
                validate_datasets.main()

    def test_result_reader_and_adapter_resume_remain_verbatim(self):
        record = normalize_record(compact())
        results = [{"task_id": record["task_id"], "target": "a", "correct": True,
                    "methods": {"soft": {"prediction": "A"}}}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            original = json.dumps(results[0]) + "\n"
            path.write_text(original)
            self.assertEqual(run_direct_reuse.load_result_jsonl(path), results)
            self.assertEqual(adapter_eval.load_completed(path, [(record, "a")]), results)
            self.assertEqual(path.read_text(), original)
            path.write_text(original + '{"task_id":')
            self.assertEqual(adapter_eval.load_completed(path, [(record, "a")]), results)
            self.assertEqual(path.read_text(), original)

    def test_truncation_archive_preserves_result_rows(self):
        outcome = {"hit_max_new_tokens": True, "generation_max_new_tokens": 4}
        truncated = {"task_id": "old-id", "full": {"a": outcome}, "reuse": {}}
        retained = {"task_id": "other-id", "full": {}, "reuse": {}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            rows, ids = run_direct_reuse.prepare_truncation_retries(path, [truncated, retained])
            self.assertEqual(rows, [retained])
            self.assertEqual(ids, {"old-id"})
            run_direct_reuse.prepare_truncation_retries(path, [truncated, retained])
            self.assertEqual(run_direct_reuse.load_result_jsonl(path.with_name("truncated_samples.jsonl")),
                             [truncated])

    def test_eval_merge_and_baseline_resume_use_raw_reader(self):
        # Verify the actual call sites, not just a standalone result reader.
        for relative, expected_count in (("latent_remedy/eval_universal_soft_suffix.py", 1),
                                         ("scripts/run_kv_boundary_calibration.py", 3)):
            tree = ast.parse((ROOT / relative).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "load_result_jsonl"]
            self.assertEqual(len(calls), expected_count)


if __name__ == "__main__":
    unittest.main()