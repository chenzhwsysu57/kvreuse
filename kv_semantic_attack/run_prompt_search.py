#!/usr/bin/env python3
"""Independent TPE/random search: compressed target prefix + generic instruction.

Run as python -m kv_semantic_attack.run_prompt_search --help. No API agents,
task-distribution adaptation, metadata-derived summaries or corrective KV edits.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from kv_semantic_attack.prompt_search import (
    BASELINES, PrefixCompressor, PromptConfig, digest, optimize_prompts,
    prompt_artifacts, render_prompt, search_space, select_shortest, split_records, write_json,
)
from kv_semantic_attack.synthetic_tasks import validate_generated_record

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("0.6b", "1.7b", "4b", "8b")


class QwenBackend:
    """One model instance per run; donor/prefix prefill still repeats per trial."""

    def __init__(self, *, model_name: str, batch_size: int, max_new_tokens: int):
        from kv_semantic_attack import evaluate_suffix_candidates as engine

        self.engine = engine
        batch = engine.batch
        self.model_path = Path(batch.snapshot_download(batch.base.MODEL_IDS[model_name], local_files_only=True))
        self.tokenizer = batch.AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = batch.AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True, dtype=batch.torch.bfloat16,
            device_map="cuda:0", attn_implementation="sdpa",
        ).eval()
        batch.base.assert_default_rope(self.model)
        self.options = {"batch_size": batch_size, "max_new_tokens": max_new_tokens,
                        "progress_every": 1, "empty_cache_every": 1}

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def full(self, records: list[dict]):
        try:
            return self.engine.full_once(self.model, self.tokenizer, records, **self.options)
        finally:
            self.engine.release_cuda()

    def reuse(self, records: list[dict], builder, label: str):
        try:
            return self.engine.evaluate(self.model, self.tokenizer, records, suffix="", reuse_engine="scatter",
                                        label=label, prefix_prompt_builder=builder, **self.options)
        finally:
            self.engine.release_cuda()


def check_rows(rows: list[dict], records: list[dict], method: str) -> None:
    expected = {(row["task_id"], side) for row in records for side in ("a", "b")}
    keys = [(row["task_id"], row["target"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected or any(method not in row for row in rows):
        raise ValueError(f"incomplete or duplicate {method} evaluation directions")


class CachedEvaluator:
    def __init__(self, output: Path, splits: dict[str, list[dict]], compressor: PrefixCompressor,
                 backend, contract: str):
        self.output, self.splits, self.compressor = output, splits, compressor
        self.backend, self.contract = backend, contract

    def full(self, split: str) -> dict:
        path = self.output / "evaluations" / split / "full.json"
        identity = digest([self.contract, split, "full"])
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["identity"] != identity:
                raise ValueError("Full cache identity mismatch")
        else:
            rows, seconds = self.backend.full(self.splits[split])
            check_rows(rows, self.splits[split], "full")
            report = {"identity": identity, "results": rows, "seconds": seconds}
            write_json(path, report)
        check_rows(report["results"], self.splits[split], "full")
        return report

    def evaluate(self, split: str, config: PromptConfig) -> dict:
        from kv_semantic_attack.eval_synthetic_batch import summarize

        records = self.splits[split]
        artifacts = prompt_artifacts(records, config, self.compressor, self.backend.token_count)
        rendered = {key: item["bridge"] for key, item in artifacts["texts_by_prefix_sha256"].items()}
        # Different parameter configurations rendering identical text share an
        # expensive evaluation. Config-specific compression metadata stays below.
        identity = digest([self.contract, split, rendered])
        path = self.output / "evaluations" / split / f"reuse_{identity}.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["identity"] != identity:
                raise ValueError("Reuse cache identity mismatch")
        else:
            full = self.full(split)["results"]
            rows, timing = self.backend.reuse(
                records, lambda prefix: render_prompt(prefix, config, self.compressor),
                f"{split}:{config.key[:12]}",
            )
            check_rows(rows, records, "reuse")
            indexed = {(row["task_id"], row["target"]): row for row in rows}
            joined = [{**row, "reuse": indexed[(row["task_id"], row["target"])]["reuse"]} for row in full]
            report = {"identity": identity, "summary": summarize(joined), "results": joined,
                      "timing": timing, "rendered_prompts": rendered}
            write_json(path, report)
        check_rows(report["results"], records, "reuse")
        return {"config": asdict(config), "accuracy": report["summary"]["overall"]["reuse_accuracy"],
                "mean_bridge_tokens": artifacts["mean_bridge_tokens"], "prompt_artifacts": artifacts,
                "summary": report["summary"], "evaluation_file": str(path.relative_to(self.output))}


def run_experiment(output: Path, splits: dict[str, list[dict]], compressor: PrefixCompressor,
                   backend, *, contract: str, budget: int, seed: int, sampler: str,
                   shortlist_size: int, tolerance_pp: float) -> dict:
    """Search -> validation selection -> frozen test. Caller owns run lock."""
    import optuna

    summary_path = output / "summary.json"
    if summary_path.exists():
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        if result["contract"] != contract:
            raise ValueError("completed run contract mismatch")
        return result
    evaluate = CachedEvaluator(output, splits, compressor, backend, contract)
    study = optuna.create_study(storage="sqlite:///" + str((output / "study.sqlite3").resolve()),
                                study_name="prefix_prompt_search", direction="maximize", load_if_exists=True)
    if study.user_attrs.get("contract", contract) != contract:
        raise ValueError("study contract mismatch")
    study.set_user_attr("contract", contract)
    baseline_search = {name: evaluate.evaluate("search", config) for name, config in BASELINES.items()}
    trials = optimize_prompts(study, compressor, lambda config: evaluate.evaluate("search", config),
                              budget=budget, seed=seed, sampler=sampler)
    ordered = sorted(trials, key=lambda row: (-row["accuracy"], row["mean_bridge_tokens"], digest(row["config"])))
    shortlist = {PromptConfig(**row["config"]).key: PromptConfig(**row["config"])
                 for row in ordered[:shortlist_size]}
    # Always compare to the complete-prefix fallback on validation, even if its
    # search score is poor. No generic-only/no-bridge candidate can win search.
    for name in ("full_generic", "compressed_generic"):
        shortlist[BASELINES[name].key] = BASELINES[name]
    validation = [evaluate.evaluate("validation", config) for config in shortlist.values()]
    selected = select_shortest(validation, tolerance_pp)
    # Persist the decision BEFORE reading any test scores; resume must preserve it.
    selection = {"contract": contract, "selected": selected, "validation": validation,
                 "accuracy_tolerance_pp": tolerance_pp,
                 "selection_rule": "shortest within tolerance of best validation accuracy in shortlist"}
    selection_path = output / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise ValueError("frozen selection changed on resume")
    write_json(selection_path, selection)
    baseline_validation = {name: evaluate.evaluate("validation", config) for name, config in BASELINES.items()}
    baseline_test = {name: evaluate.evaluate("test", config) for name, config in BASELINES.items()}
    test = evaluate.evaluate("test", PromptConfig(**selected["config"]))
    result = {"complete": True, "contract": contract, "sampler": sampler, "search": trials,
              "selection": selection, "test": test,
              "baselines": {"search": baseline_search, "validation": baseline_validation, "test": baseline_test},
              "test_improvement_over_direct_pp": 100 * (test["accuracy"] - baseline_test["direct_reuse"]["accuracy"]),
              "note": "Test was not used for candidate selection. Full is the original, unmodified prompt baseline."}
    write_json(summary_path, result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="validated synthetic paired JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampler", choices=("tpe", "random"), default="tpe")
    parser.add_argument("--trials", type=int, default=16, help="total distinct configurations, including warm starts")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--model", choices=MODEL_NAMES, default="1.7b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--test-fraction", type=float, default=.2)
    parser.add_argument("--shortlist-size", type=int, default=5)
    parser.add_argument("--accuracy-tolerance-pp", type=float, default=0.0,
                        help="validation accuracy loss allowed in return for a shorter bridge; default no loss")
    parser.add_argument("--compression-bank", type=Path, help="optional prefix-only semantic compression variants")
    parser.add_argument("--dry-run", action="store_true", help="validate data and preview prompts; no model, API, or output writes")
    args = parser.parse_args(argv)
    if min(args.trials, args.batch_size, args.max_new_tokens, args.shortlist_size) < 1:
        parser.error("trial, batch, token and shortlist counts must be positive")
    if not math.isfinite(args.accuracy_tolerance_pp) or not 0 <= args.accuracy_tolerance_pp <= 100:
        parser.error("accuracy tolerance must be between 0 and 100 pp")
    return args


def source_fingerprint() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "kv_semantic_attack" / "prompt_search.py",
             ROOT / "kv_semantic_attack" / "eval_synthetic_batch.py",
             ROOT / "kv_semantic_attack" / "evaluate_suffix_candidates.py",
             ROOT / "kv_semantic_attack" / "batched_reuse.py", ROOT / "kv_semantic_attack" / "synthetic_tasks.py",
             ROOT / "scripts" / "batch_eval.py", ROOT / "scripts" / "run_direct_reuse.py"]
    return {str(path.relative_to(ROOT)): digest(path.read_text(encoding="utf-8")) for path in paths}


def main(argv=None) -> int:
    args = parse_args(argv)
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        validate_generated_record(record)
    bank = {}
    if args.compression_bank:
        bank = json.loads(args.compression_bank.read_text(encoding="utf-8"))["variants"]
        if not isinstance(bank, dict):
            raise ValueError("compression bank 'variants' must be an object")
    compressor = PrefixCompressor(bank)
    if args.trials > len(search_space(compressor)):
        raise ValueError(f"only {len(search_space(compressor))} configurations in this search space")
    splits = split_records(records, seed=args.seed, validation_fraction=args.validation_fraction,
                           test_fraction=args.test_fraction)
    plan = {"input_sha256": digest(records), "seed": args.seed,
            "splits": {name: {"pairs": len(rows), "groups": len({row["shared_data_id"] for row in rows}),
                              "task_types": dict(Counter(row["attack_type"] for row in rows)),
                              "task_ids": [row["task_id"] for row in rows]} for name, rows in splits.items()},
            "search_space_size": len(search_space(compressor)), "compression_modes": list(compressor.modes),
            "placement": "original target prefix -> reused block KV -> restatement + generic prompt -> question"}
    if args.dry_run:
        prefix = splits["search"][0]["prefix_a"]
        preview = {mode: render_prompt(prefix, PromptConfig(compression=mode), compressor) for mode in compressor.modes}
        print(json.dumps({"dry_run": True, **plan, "original_prefix": prefix, "preview": preview}, ensure_ascii=False, indent=2))
        return 0

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process owns this output directory") from exc
        # Resolve the LOCAL model before inference and record a lightweight file
        # identity (size/mtime, not a costly full multi-GB weights checksum).
        from kv_semantic_attack import evaluate_suffix_candidates as engine

        model_path = Path(engine.batch.snapshot_download(engine.batch.base.MODEL_IDS[args.model], local_files_only=True))
        model_files = {str(path.relative_to(model_path)): [path.stat().st_size, path.stat().st_mtime_ns]
                       for path in sorted(model_path.rglob("*")) if path.is_file()}
        options = {key: value for key, value in vars(args).items()
                   if key not in {"input", "output_dir", "compression_bank", "dry_run"}}
        runtime = {name: importlib.metadata.version(name) for name in ("optuna", "torch", "transformers", "modelscope")}
        runtime["cuda"] = engine.batch.torch.version.cuda
        runtime["gpu"] = engine.batch.torch.cuda.get_device_name(0)
        manifest = {"version": 1, "plan": plan, "options": options, "bank_sha256": digest(bank),
                    "sources": source_fingerprint(), "packages": runtime,
                    "model_path": str(model_path.resolve()), "model_files": model_files}
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("run configuration/data/code/model changed; use a new output directory")
        elif any(path.name != ".run.lock" for path in output.iterdir()):
            raise ValueError("refusing a nonempty output directory without a matching manifest")
        else:
            write_json(manifest_path, manifest)
        contract = digest(manifest)
        if (output / "summary.json").exists():
            print(f"Already complete: {output / 'summary.json'}")
            return 0
        for name, rows in splits.items():
            write_json(output / "splits" / f"{name}.json", rows)
        backend = QwenBackend(model_name=args.model, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
        result = run_experiment(output, splits, compressor, backend, contract=contract, budget=args.trials,
                                seed=args.seed, sampler=args.sampler, shortlist_size=args.shortlist_size,
                                tolerance_pp=args.accuracy_tolerance_pp)
        print(json.dumps({"selected_config": result["test"]["config"], "test_accuracy": result["test"]["accuracy"],
                          "mean_bridge_tokens": result["test"]["mean_bridge_tokens"],
                          "report": str(output / "summary.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())