"""CPU-only tests for the public CLI; no model or tensor imports required."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_run", ROOT / "run.py")
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class PublicRunTests(unittest.TestCase):
    def record(self, task_id: str):
        return {"task_id": task_id, "dataset": "custom", "prefix_a": "Role A",
                "prefix_b": "Role B", "shared_block": "[A] first\n[B] second",
                "question": "Choose one.", "gold_a": "A", "gold_b": "B",
                "metric": "exact_match"}

    def parse(self, *selection: str, reasoning: str = "no", **environment: str):
        argv = ["run.py", *selection, "--method", "reuse", "--reasoning", reasoning,
                "--kv-mode", "a2b"]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, environment, clear=True):
            return run.parse_args()

    def test_registered_datasets_exist_and_match_counts(self):
        expected = {"argkp": 110, "deal": 110, "job": 110, "all330": 330,
                    "all240": 7200, "train180": 5400, "test60": 1800}
        self.assertEqual(set(run.DATASETS), set(expected))
        with tempfile.TemporaryDirectory() as temporary:
            for label, count in expected.items():
                with self.subTest(dataset=label):
                    relative, dataset_name, declared = run.DATASETS[label]
                    self.assertTrue((ROOT / relative).is_file())
                    self.assertEqual(declared, count)
                    args = self.parse("--dataset", label)
                    snapshot, rows, sources = run.select_input(args, Path(temporary) / label)
                    self.assertEqual(len(rows), count)
                    self.assertEqual(len(run.load_jsonl(snapshot)), count)
                    self.assertEqual(sources, [ROOT / relative])
                    if dataset_name is not None:
                        self.assertEqual({row["dataset"] for row in rows}, {dataset_name})

    def test_selection_is_exactly_one_and_label_requires_input(self):
        invalid = [(), ("--dataset", "argkp", "--input", "missing.jsonl"),
                   ("--dataset", "argkp", "--input-label", "custom"),
                   ("--dataset", "synthetic16-all"), ("--input", "missing.jsonl")]
        for selection in invalid:
            with self.subTest(selection=selection), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse(*selection)

    def test_custom_input_and_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "custom.jsonl"
            rows = [self.record(f"custom-{index}") for index in range(3)]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            args = self.parse("--input", str(source), "--input-label", "My-input", "--limit", "2")
            snapshot, selected, sources = run.select_input(args, Path(temporary) / "output")
            self.assertIsNone(args.dataset)
            self.assertEqual(args.input_label, "My-input")
            self.assertEqual(selected, rows[:2])
            self.assertEqual(run.load_jsonl(snapshot), rows[:2])
            self.assertEqual(sources, [source.resolve()])

    def test_empty_custom_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "empty.jsonl"
            source.touch()
            args = self.parse("--input", str(source))
            with self.assertRaises(ValueError):
                run.select_input(args, Path(temporary) / "output")

    def test_custom_labels_and_output_are_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "odd name.jsonl"
            source.write_text('{"task_id": "one"}\n', encoding="utf-8")
            for label in ("../escape", "/absolute", ".", "..", "bad/name", "", "a" * 65):
                with self.subTest(label=label), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self.parse("--input", str(source), "--input-label", label)
            args = self.parse("--input", str(source), "--output-root", str(root / "results"))
            self.assertEqual(args.input_label, "odd-name")
            directory = run.run_directory(args, "reuse")
            self.assertTrue(directory.is_relative_to(root / "results"))
            self.assertTrue(directory.relative_to(root / "results").parts[0].startswith("custom-odd-name-"))
            self.assertNotEqual(run.input_output_label(args), "argkp")
            other = root / "other.jsonl"
            other.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            other_args = self.parse("--input", str(other), "--input-label", args.input_label)
            self.assertNotEqual(run.input_output_label(args), run.input_output_label(other_args))
            # A malicious programmatic label must not bypass the directory guard.
            args.input_label = "../../../../escape"
            with self.assertRaises(ValueError):
                run.run_directory(args, "reuse")

    def test_dry_run_preserves_input_and_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "custom.jsonl"
            content = json.dumps(self.record("one")) + "\n"
            source.write_text(content, encoding="utf-8")
            argv = ["run.py", "--input", str(source), "--method", "reuse", "--reasoning", "yes",
                    "--kv-mode", "a2b", "--output-root", str(root / "results"), "--dry-run"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(), 0)
                with self.assertRaises(FileExistsError):
                    run.main()
            manifests = list((root / "results").rglob("run_config.json"))
            self.assertEqual(len(manifests), 1)
            config = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(config["custom_input"], str(source))
            self.assertEqual(config["records"], 1)
            self.assertEqual(source.read_text(encoding="utf-8"), content)

    def test_portable_runtime_defaults_and_environment_overrides(self):
        args = self.parse("--dataset", "argkp")
        self.assertEqual(args.relay_python, Path(sys.executable))
        self.assertIsNone(args.torch_runtime)
        self.assertEqual(args.output_root, ROOT / "results/unified")
        args = self.parse("--dataset", "argkp", RELAY_PYTHON="relay-python",
                          KVREUSE_TORCH_RUNTIME="explicit-runtime")
        self.assertEqual(args.relay_python, Path("relay-python"))
        self.assertEqual(args.torch_runtime, Path("explicit-runtime"))

    def test_commands_reasoning_is_only_explicit_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "input.jsonl"
            input_path.write_text(json.dumps(self.record("one")) + "\n", encoding="utf-8")
            checkpoint = directory / "checkpoint.pt"
            checkpoint.touch()
            for method in sorted(set(run.METHOD_ALIASES.values())):
                for reasoning in ("no", "yes"):
                    with self.subTest(method=method, reasoning=reasoning):
                        args = self.parse("--dataset", "argkp", reasoning=reasoning)
                        args.max_new_tokens = 128
                        args.baseline = input_path
                        args.soft_prompt_checkpoint = checkpoint
                        sides, directions = run.modes(method, "all")
                        command = run.build_command(args, method, input_path, directory, sides, directions)
                        self.assertFalse(any("thinking" in flag for flag in command))
                        if method == "universal-soft-suffix":
                            self.assertEqual(command[2],
                                             str(ROOT / "latent_remedy/eval_universal_soft_suffix.py"))
                            self.assertTrue(Path(command[2]).is_file())
                            self.assertEqual(command[command.index("--eval-mode") + 1],
                                             "reasoning" if reasoning == "yes" else "no-reasoning")
                        else:
                            self.assertEqual("--explicit-reasoning" in command, reasoning == "yes")
                        self.assertFalse(any("accbaseline" in flag for flag in command))

    def test_prompt_builders_keep_thinking_disabled(self):
        # Inspect real prompt calls without importing GPU-dependent modules.
        paths = ["scripts/batch_eval.py", "latent_remedy/train_universal_soft_suffix.py",
                 "latent_remedy/eval_universal_soft_suffix.py",
                 "latent_remedy/train_prefix_adapter_repeat.py",
                 "latent_remedy/eval_prefix_adapter_repeat.py"]
        for relative in paths:
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "build_prompt_parts"]
            with self.subTest(path=relative):
                self.assertTrue(calls)
                for call in calls:
                    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
                    self.assertIsInstance(keywords["enable_thinking"], ast.Constant)
                    self.assertIs(keywords["enable_thinking"].value, False)
                    self.assertIn("explicit_reasoning", keywords)


    def test_package_import_and_wrapper_delegation(self):
        spec = importlib.util.spec_from_file_location("latent_remedy", ROOT / "latent_remedy/__init__.py")
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(package)
        self.assertEqual(package.__name__, "latent_remedy")
        self.assertEqual(Path(package.__file__).resolve(), ROOT / "latent_remedy/__init__.py")
        self.assertEqual(list(package.__path__), [str(ROOT / "latent_remedy")])
        tree = ast.parse((ROOT / "train_prefix_adapter_repeat.py").read_text(encoding="utf-8"))
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "latent_remedy.train_prefix_adapter_repeat")
        self.assertEqual([alias.name for alias in imports[0].names], ["main"])
        self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()