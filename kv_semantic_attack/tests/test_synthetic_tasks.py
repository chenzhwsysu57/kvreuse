"""Offline correctness tests; no API, torch, tokenizer or GPU required."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from kv_semantic_attack.synthetic_tasks import (
    DIRECT_OPTION_QUESTION, LAYOUTS, TASK_TYPES, generate_tasks, render_block, score_response, solve,
    validate_generated_record,
)

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "kv_semantic_attack/generate_synthetic_tasks.py"


def parse_shared_block(block, layout, kind):
    """Independent reader of what the model sees, not the stored gold data."""
    body, options = block.split("\n\nOptions:\n", 1)
    option_rows = [line.split(". ", 1) for line in options.splitlines()]
    option_letters, option_values = zip(*option_rows)
    option_values = [json.loads(value) for value in option_values]
    if kind == "expression":
        return {"kind": kind, "numbers": [int(x) for x in body.removeprefix("Expression: ").split(" + ")],
                "option_letters": list(option_letters), "option_values": list(option_values)}
    if kind == "sets":
        left, right = body.splitlines()
        left_name, left_items = left.split(": ", 1)
        right_name, right_items = right.split(": ", 1)
        return {"kind": kind, "left_name": left_name, "right_name": right_name,
                "left": left_items.split(", "), "right": right_items.split(", "),
                "option_letters": list(option_letters), "option_values": list(option_values)}
    if layout == "json":
        rows = json.loads(body)
    elif layout == "table":
        lines = body.splitlines()
        columns = lines[0].split(" | ")
        rows = [dict(zip(columns, line.split(" | "))) for line in lines[1:]]
    else:
        rows = [dict(part.split("=", 1) for part in line.split("; ")) for line in body.splitlines()]
    for row in rows:
        for key in ("value", "cost", "speed"):
            if key in row:
                row[key] = int(row[key])
        for key in ("p", "q", "active"):
            if key in row and isinstance(row[key], str):
                row[key] = row[key] == "True"
    return {"kind": "rows", "rows": rows,
            "option_letters": list(option_letters), "option_values": list(option_values)}


class GeneratedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = list(generate_tasks({name: 100 for name in TASK_TYPES}, seed=35))

    def test_exact_counts_ids_and_conflicts(self):
        self.assertEqual(len(self.records), 100 * len(TASK_TYPES))
        self.assertEqual(Counter(r["attack_type"] for r in self.records), {name: 100 for name in TASK_TYPES})
        self.assertEqual(len({r["task_id"] for r in self.records}), len(self.records))
        self.assertEqual(len({r["shared_data_id"] for r in self.records}), len(self.records))
        for record in self.records:
            validate_generated_record(record)
            self.assertNotEqual(record["gold_a"], record["gold_b"])
            self.assertIn(record["gold_a"], "ABCD")
            self.assertIn(record["gold_b"], "ABCD")
            self.assertNotEqual(record["prefix_a"], record["prefix_b"])
            self.assertEqual(record["answer_mode"], "raw")

    def test_visible_material_matches_oracle_data(self):
        for record in self.records:
            meta = record["metadata"]
            reconstructed = parse_shared_block(record["shared_block"], meta["layout"], meta["data"]["kind"])
            self.assertEqual(reconstructed, meta["data"])
            for side in ("a", "b"):
                answer = solve(reconstructed, meta[f"rule_{side}"])
                expected = reconstructed["option_letters"][reconstructed["option_values"].index(answer)]
                self.assertEqual(expected, record[f"gold_{side}"])

    def test_deterministic_and_independent_of_counts(self):
        self.assertEqual(self.records, list(generate_tasks({name: 100 for name in TASK_TYPES}, seed=35)))
        subset = list(generate_tasks({"priority_switch": 5}, seed=35))
        expected = [r for r in self.records if r["attack_type"] == "priority_switch"][:5]
        self.assertEqual(subset, expected)
        self.assertNotEqual(subset, list(generate_tasks({"priority_switch": 5}, seed=36)))

    def test_diversity_and_real_priority_ties(self):
        for task_type in TASK_TYPES:
            rows = [r for r in self.records if r["attack_type"] == task_type]
            self.assertGreater(len({r["metadata"]["instruction_style"] for r in rows}), 1)
            self.assertGreater(len({r["metadata"]["num_rows"] for r in rows}), 1)
            if task_type not in {"task_switch", "set_relation"}:
                self.assertEqual({r["metadata"]["layout"] for r in rows}, set(LAYOUTS))
            if task_type == "priority_switch":
                for record in rows:
                    meta = record["metadata"]
                    for side in ("a", "b"):
                        field = meta[f"rule_{side}"]["fields"][0]
                        values = [row[field] for row in meta["data"]["rows"]]
                        self.assertEqual(values.count(min(values)), 2)

    def test_raw_scoring_preserves_constraints(self):
        for record in self.records:
            for side in ("a", "b"):
                gold = record[f"gold_{side}"]
                self.assertTrue(score_response(record, side, "\n" + gold + "\n"))
                self.assertFalse(score_response(record, side, f"The answer is {gold}"))
                self.assertFalse(score_response(record, side, f"\\boxed{{{gold}}}"))
                other = "b" if side == "a" else "a"
                self.assertFalse(score_response(record, side, record[f"gold_{other}"]))
        record = next(r for r in self.records if r["attack_type"] == "case_switch")
        self.assertFalse(score_response(record, "a", record["gold_a"].lower()))
        with self.assertRaises(ValueError):
            score_response(record, "source", "A")

    def test_corruption_is_rejected(self):
        for key in ("gold_a", "gold_b", "shared_block", "prefix_a", "question", "shared_data_id"):
            record = copy.deepcopy(self.records[0])
            record[key] += "corrupt"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_generated_record(record)

    def test_bounds_and_unknown_types(self):
        invalid = [
            ({"missing": 1}, {}), ({"field_switch": -1}, {}),
            ({"field_switch": 1.5}, {}), ({"field_switch": True}, {}),
            ({"field_switch": 1}, {"min_rows": 3}),
            ({"field_switch": 1}, {"max_rows": 101}),
            ({"field_switch": 1}, {"layouts": ()}),
            ({"field_switch": 1}, {"layouts": ("invalid",)}),
            ({"scope_switch": 1}, {"min_rows": 5, "max_rows": 5}),
        ]
        for counts, kwargs in invalid:
            with self.subTest(counts=counts, kwargs=kwargs), self.assertRaises(ValueError):
                list(generate_tasks(counts, **kwargs))
        self.assertEqual(list(generate_tasks({"field_switch": 0})), [])

    def test_many_seeds_and_fixed_layouts(self):
        for seed in range(10):
            records = list(generate_tasks({name: 10 for name in TASK_TYPES}, seed=seed, min_rows=4, max_rows=4, layouts=("json",)))
            self.assertEqual(len(records), 10 * len(TASK_TYPES))
            for record in records:
                validate_generated_record(record)
                self.assertEqual(record["metadata"]["num_rows"], 4)

    def test_direct_option_format_control_has_no_raw_output_instruction(self):
        records = list(generate_tasks({"format_switch": 20}, seed=81, prompt_protocol="direct_option_format"))
        self.assertEqual(len(records), 20)
        for record in records:
            validate_generated_record(record)
            self.assertEqual(record["question"], DIRECT_OPTION_QUESTION)
            self.assertEqual(record["metadata"]["prompt_protocol"], "direct_option_format")
            for side in ("a", "b"):
                prefix = record[f"prefix_{side}"]
                self.assertIn("select the option whose payload", prefix.lower())
                self.assertIn("return only its option letter", prefix.lower())
                self.assertNotIn("output a compact", prefix.lower())
                self.assertNotIn("output exactly two", prefix.lower())
        with self.assertRaises(ValueError):
            list(generate_tasks({"format_switch": 1, "field_switch": 1}, prompt_protocol="direct_option_format"))


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.data = {"kind": "rows", "rows": [
            {"id": "R1", "value": 4, "color": "red", "cost": 1, "speed": 4, "name": "alice"},
            {"id": "R2", "value": 9, "color": "blue", "cost": 1, "speed": 3, "name": "brian"},
            {"id": "R3", "value": 2, "color": "red", "cost": 5, "speed": 1, "name": "carol"},
            {"id": "R4", "value": 7, "color": "blue", "cost": 4, "speed": 1, "name": "david"},
        ], "option_letters": ["A", "B", "C", "D"],
        "option_values": ["R1", "R2", "R3", "R4"]}

    def test_selection_and_priority(self):
        rules = [
            ({"op": "select", "fields": ["value"], "order": "max"}, "R2"),
            ({"op": "select", "fields": ["value"], "order": "min"}, "R3"),
            ({"op": "select", "fields": ["value"], "order": "max", "filter": ["color", "red"]}, "R1"),
            ({"op": "select", "fields": ["value"], "order": "max", "scope": "second"}, "R4"),
            ({"op": "select", "fields": ["cost", "speed"], "order": "min"}, "R2"),
            ({"op": "select", "fields": ["speed", "cost"], "order": "min"}, "R4"),
        ]
        for rule, gold in rules:
            self.assertEqual(solve(self.data, rule), gold)
        with self.assertRaises(ValueError):
            solve(self.data, {"op": "select", "fields": ["cost"], "order": "min"})
        with self.assertRaises(ValueError):
            solve(self.data, {"op": "select", "fields": ["value"], "order": "max", "filter": ["color", "green"]})

    def test_operations(self):
        self.assertEqual(solve(self.data, {"op": "aggregate", "field": "value", "mode": "sum"}), "22")
        self.assertEqual(solve(self.data, {"op": "aggregate", "mode": "count"}), "4")
        self.assertEqual(solve(self.data, {"op": "sort", "field": "value", "descending": False}), "R3,R1,R4,R2")
        self.assertEqual(solve(self.data, {"op": "sort", "field": "value", "descending": True}), "R2,R4,R1,R3")
        expression = {"kind": "expression", "numbers": [8, 2, 8],
                  "option_letters": ["A", "B", "C", "D"], "option_values": ["1", "2", "3", "4"]}
        self.assertEqual(solve(expression, {"op": "sum_numbers"}), "18")
        self.assertEqual(solve(expression, {"op": "extract_numbers"}), "8,2,8")

    def test_encoding(self):
        rule = {"op": "label", "target_id": "R1", "threshold": 4, "true_label": "A", "false_label": "B"}
        self.assertEqual(solve(self.data, rule), "A")
        self.assertEqual(solve(self.data, {**rule, "threshold": 5}), "B")
        self.assertEqual(solve(self.data, {"op": "format", "target_id": "R1", "format": "json"}), '{"id":"R1","value":4}')
        self.assertEqual(solve(self.data, {"op": "format", "target_id": "R1", "format": "csv"}), "id,value\nR1,4")
        self.assertEqual(solve(self.data, {"op": "case", "target_id": "R1", "case": "upper"}), "ALICE")
        self.assertEqual(solve(self.data, {"op": "case", "target_id": "R1", "case": "lower"}), "alice")

    def test_layout_roundtrip(self):
        for layout in LAYOUTS:
            self.assertEqual(parse_shared_block(render_block(self.data, layout), layout, "rows"), self.data)


class CLITests(unittest.TestCase):
    def run_cli(self, output, *args):
        return subprocess.run([sys.executable, str(CLI), "--output-dir", str(output), *args],
                              cwd="/tmp", capture_output=True, text=True, check=False)

    def test_counts_manifest_and_protected_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "generated"
            args = ("--per-type", "2", "--count", "label_mapping=5")
            result = self.run_cli(output, *args)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["pairs"], 2 * len(TASK_TYPES) + 3)
            self.assertEqual(manifest["reuse_directions"], 2 * manifest["pairs"])
            original = (output / "all.jsonl").read_bytes()
            self.assertEqual(len(original.splitlines()), manifest["pairs"])
            self.assertNotEqual(self.run_cli(output, *args).returncode, 0)
            self.assertEqual((output / "all.jsonl").read_bytes(), original)
            self.assertEqual(self.run_cli(output, *args, "--overwrite").returncode, 0)
            self.assertEqual((output / "all.jsonl").read_bytes(), original)
            (output / "unrelated.txt").write_text("keep")
            self.assertNotEqual(self.run_cli(output, *args, "--overwrite").returncode, 0)
            self.assertEqual((output / "unrelated.txt").read_text(), "keep")

    def test_subset_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "subset"
            result = self.run_cli(output, "--types", "format_switch", "--per-type", "3")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len((output / "all.jsonl").read_text().splitlines()), 3)
            self.assertFalse((output / "field_switch.jsonl").exists())
            for args in (("--per-type", "-1"), ("--count", "bad=10"), ("--per-type", "0"),
                         ("--types", "scope_switch", "--min-rows", "5", "--max-rows", "5")):
                result = self.run_cli(Path(tmp) / "invalid", *args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((Path(tmp) / "invalid").exists())


if __name__ == "__main__":
    unittest.main()