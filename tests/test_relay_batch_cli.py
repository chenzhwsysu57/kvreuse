"""CPU-only Relay CLI integration tests; no model/tensor imports."""
from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import argparse

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("relay_public_run", ROOT / "run.py")
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class RelayBatchCLITests(unittest.TestCase):
    def parse(self, *flags, method="epic", reasoning="no"):
        argv = ["run.py", "--dataset", "all330", "--method", method,
                "--reasoning", reasoning, "--kv-mode", "all", *flags]
        with mock.patch.object(sys, "argv", argv):
            return run.parse_args()

    def command(self, args, method):
        with mock.patch.object(run, "load_jsonl", return_value=[{}, {}]):
            return run.build_command(args, method, ROOT / "input.jsonl", ROOT / "results/mock",
                                     ["b", "a"], ["a_to_b", "b_to_a"])

    def test_official_defaults_and_generation_budget(self):
        for method in sorted(run.RELAY_METHODS):
            for reasoning, budget in (("no", "128"), ("yes", "4096")):
                with self.subTest(method=method, reasoning=reasoning):
                    args = self.parse(method=method, reasoning=reasoning)
                    command = self.command(args, method)
                    for flag, value in (("--execution-path", "official-serial"),
                                        ("--batch-size", "1"), ("--warmup-batches", "0"),
                                        ("--epic-prefix-tokens", "16"), ("--max-new-tokens", budget)):
                        self.assertEqual(command[command.index(flag) + 1], value)
                    self.assertEqual("--explicit-reasoning" in command, reasoning == "yes")
                    self.assertFalse(any("thinking" in flag for flag in command))
                    self.assertNotIn("--run-mode", command)

    def test_build_command_forwards_all_batch_controls(self):
        args = self.parse("--relay-execution-path", "static-batched", "--batch-size", "3",
                          "--warmup-batches", "2", "--epic-prefix-tokens", "7",
                          "--relay-start-layer", "1", "--relay-det-layer", "2",
                          "--relay-end-layer", "5", "--relay-tau-dev", "1.5",
                          "--relay-tau-inf", "0.4", "--relay-suffix-tokens", "6",
                          "--max-new-tokens", "33", "--overwrite", reasoning="yes")
        command = self.command(args, "epic")
        expected = {"--method": "epic", "--execution-path": "static-batched",
                    "--batch-size": "3", "--warmup-batches": "2", "--epic-prefix-tokens": "7",
                    "--start-layer": "1", "--det-layer": "2", "--end-layer": "5",
                    "--tau-dev": "1.5", "--tau-inf": "0.4", "--suffix-tokens": "6",
                    "--max-new-tokens": "33", "--limit": "2"}
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(command[command.index("--directions") + 1:][:2], ["a_to_b", "b_to_a"])
        self.assertIn("--overwrite", command)
        for method in ("cacheblend", "relaycaching"):
            self.assertIn("static-batched", self.command(
                self.parse("--relay-execution-path", "static-batched", "--batch-size", "2",
                           method=method), method))

    def test_invalid_flags_fail_before_output_creation(self):
        invalid = [(("--batch-size", "0"), "epic"), (("--batch-size", "2"), "epic"),
                   (("--warmup-batches", "1"), "epic"), (("--warmup-batches", "-1"), "epic"),
                   (("--epic-prefix-tokens", "-1"), "epic"),
                   (("--relay-suffix-tokens", "-1"), "epic"),
                   (("--max-new-tokens", "0"), "epic"), (("--max-new-tokens", "-2"), "epic"),
                   (("--relay-start-layer", "-1"), "epic"),
                   (("--relay-start-layer", "2", "--relay-det-layer", "1"), "epic"),
                   (("--relay-end-layer", "-2"), "epic"),
                   (("--relay-det-layer", "2", "--relay-end-layer", "1"), "epic"),
                   (("--relay-tau-dev", "nan"), "epic"),
                   (("--relay-tau-inf", "inf"), "epic"),
                   (("--relay-tau-inf", "-0.1"), "epic"),
                   (("--epic-prefix-tokens", "8"), "cacheblend"),
                   (("--relay-execution-path", "static-batched"), "reuse"),
                   (("--warmup-batches", "1"), "repeat-prefix"),
                   (("--relay-suffix-tokens", "2"), "full-prefill")]
        for flags, method in invalid:
            with self.subTest(flags=flags, method=method), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.parse(*flags, method=method)
                self.assertEqual(error.exception.code, 2)
        args = self.parse()
        args.batch_size = 2
        with self.assertRaises(ValueError):
            self.command(args, "epic")

    def test_dry_run_config_preserves_public_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.parse("--relay-execution-path", "static-batched", "--batch-size", "2",
                              "--warmup-batches", "1", "--relay-suffix-tokens", "3",
                              "--limit", "1", "--output-root", temporary, "--dry-run")
            with mock.patch.object(run, "parse_args", return_value=args), \
                    mock.patch.object(run.subprocess, "Popen") as child, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(), 0)
                child.assert_not_called()
            path = next(Path(temporary).rglob("run_config.json"))
            config = json.loads(path.read_text())
            for key in ("relay_execution_path", "relay_start_layer", "relay_det_layer",
                        "relay_end_layer", "relay_tau_dev", "relay_tau_inf",
                        "relay_suffix_tokens", "epic_prefix_tokens", "warmup_batches"):
                self.assertEqual(config[key], getattr(args, key))
            self.assertEqual(config["records"], 1)
            self.assertEqual(config["dataset"], "all330")
            self.assertEqual(config["batch_size"], 2)
            self.assertIn("input_snapshot_sha256", config)

    def test_runner_parser_defaults_and_invalid_combinations_without_imports(self):
        # Execute only the real parser AST: do not load LLMChat, torch or models.
        tree = ast.parse((ROOT / "scripts/run_relay_methods.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
        namespace = {"argparse": argparse, "Path": Path, "ROOT": ROOT,
                     "METHODS": ("relaycaching", "cacheblend", "epic"), "MODEL_PATHS": {"4b": "mock"}}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "runner-parser", "exec"), namespace)
        with mock.patch.object(sys, "argv", ["runner", "--method", "epic", "--model", "4b"]):
            args = namespace["parse_args"]()
        self.assertEqual(args.execution_path, "official-serial")
        self.assertEqual(args.input, ROOT / run.DATASETS["all330"][0])
        self.assertTrue(args.input.is_file())
        for flags in (("--batch-size", "2"), ("--warmup-batches", "1"),
                      ("--max-new-tokens", "0"), ("--method", "cacheblend", "--epic-prefix-tokens", "8")):
            with mock.patch.object(sys, "argv", ["runner", "--method", "epic", *flags]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                namespace["parse_args"]()


if __name__ == "__main__":
    unittest.main()