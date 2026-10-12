#!/usr/bin/env python3
"""Train a latent remedy prompt with an explicit, reproducible task selection.

Run this script from an activated ``kvreuse`` environment.  It selects synthetic
training task families from a direct-reuse evaluation summary (lowest accuracy
first), or uses task names supplied explicitly, then delegates optimization to
``latent_remedy/train_universal_soft_suffix.py``.

Examples
--------
# Pick the three synthetic task families with lowest direct-reuse accuracy.
python train-latent-token.py \
  --ranking-summary results/universal_soft_suffix_train3_synthetic300_small5_all_holdout_4b_20260918/summary.json \
  --num-tasks 3 --samples-per-task 5 --teacher-tokens 8 --soft-tokens 64 \
  --model 4b --output results/latent_ranked3_pairs5_soft64_teacher8_4b

# Use an explicitly declared task set rather than ranking selection.
python train-latent-token.py \
  --task synthetic_boolean_logic --task synthetic_set_relation \
  --samples-per-task 10 --teacher-tokens 3 --soft-tokens 32 --model 4b \
  --output results/latent_manual2_pairs10_soft32_teacher3_4b
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data/benchmark/generated240/full_240classes_30each.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="JSONL containing candidate synthetic task families")
    parser.add_argument("--extra-input", type=Path, action="append", default=[],
                        help="additional JSONL pool whose task families may be selected; repeat as needed")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ranking-summary", type=Path,
                           help="evaluation summary.json used to rank task families by Direct accuracy")
    selection.add_argument("--task", action="append",
                           help="explicit dataset/task-family name; repeat for every training task")
    parser.add_argument("--num-tasks", type=int,
                        help="number of lowest-Direct task families selected from --ranking-summary")
    parser.add_argument("--samples-per-task", type=int, required=True,
                        help="number of disjoint source groups sampled for training in every selected task")
    parser.add_argument("--teacher-tokens", type=int, required=True,
                        help="number of greedy Full teacher tokens distilled per direction")
    parser.add_argument("--soft-tokens", type=int, required=True,
                        help="number of learned latent remedy tokens")
    parser.add_argument("--train-batch-size", type=int, default=8,
                        help="cached training directions per optimizer update")
    parser.add_argument("--model", choices=("0.6b", "1.7b", "4b", "8b"), default="4b")
    parser.add_argument("--reasoning", choices=("no", "yes", "mixed"), default="no",
                        help="distill answer-only, explicit-reasoning, or both prompt formats into one soft prompt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--eval-directions", type=int, default=0,
                        help="held-out KL directions during training; default 0 keeps the hold-out untouched")
    parser.add_argument("--generate-directions", type=int, default=0,
                        help="held-out generation directions after training")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--eval-input", type=Path, action="append",
                        help="JSONL evaluated after training; repeat for multiple unseen task pools. Defaults to --input")
    parser.add_argument("--eval-mode", choices=("no-reasoning", "reasoning"), default="no-reasoning",
                        help="evaluate the learned tokens without or with explicit reasoning")
    parser.add_argument("--eval-max-new-tokens", type=int,
                        help="generation budget for unseen-task evaluation; defaults to 32/no-reasoning or 4096/reasoning")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="evaluate unseen tasks every N training steps; 0 disables early stopping")
    parser.add_argument("--eval-start-step", type=int, default=1,
                        help="first training step eligible for unseen-task evaluation")
    parser.add_argument("--eval-patience", type=int, default=500,
                        help="stop after this many steps without a better unseen-task Soft accuracy; 0 disables")
    parser.add_argument("--eval-batch-size", type=int, default=24,
                        help="directions per batched unseen-task evaluation forward")
    parser.add_argument("--eval-pairs-per-task", type=int, default=10,
                        help="held-out pairs per selected training task for periodic evaluation")
    parser.add_argument("--periodic-eval-kind", choices=("kl", "generation"), default="kl",
                        help="use fast teacher-forced KL (default) or free-generation accuracy during training")
    parser.add_argument("--skip-unseen-eval", action="store_true",
                        help="do not run the default post-training evaluation on task families unseen in training")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--torch-runtime", type=Path,
                        default=Path(os.environ["KVREUSE_TORCH_RUNTIME"])
                        if os.environ.get("KVREUSE_TORCH_RUNTIME") else None,
                        help="optional torch/nvidia/triton packages; set explicitly or via KVREUSE_TORCH_RUNTIME")
    parser.add_argument("--overwrite", action="store_true",
                        help="remove an existing --output directory before training")
    parser.add_argument("--save-final", action="store_true",
                        help="save final_latent_tokens.pt after training, even without periodic evaluation")
    parser.add_argument("--dry-run", action="store_true",
                        help="print selected tasks and delegated command without training")
    args = parser.parse_args()
    if args.num_tasks is not None and args.num_tasks < 1:
        parser.error("--num-tasks must be positive")
    if args.ranking_summary and args.num_tasks is None:
        parser.error("--ranking-summary requires --num-tasks")
    if args.task and args.num_tasks is not None:
        parser.error("--num-tasks is only valid with --ranking-summary")
    for name in ("samples_per_task", "teacher_tokens", "soft_tokens", "train_batch_size", "steps", "max_new_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.eval_max_new_tokens is not None and args.eval_max_new_tokens < 1:
        parser.error("--eval-max-new-tokens must be positive")
    if args.eval_every < 0 or args.eval_patience < 0:
        parser.error("--eval-every and --eval-patience must be non-negative")
    if args.eval_start_step < 1:
        parser.error("--eval-start-step must be positive")
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    if args.eval_pairs_per_task < 1:
        parser.error("--eval-pairs-per-task must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def load_candidate_datasets(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"input does not exist: {path}")
    datasets = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        dataset = row.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"{path}:{line_number}: missing dataset")
        datasets.add(dataset)
    if not datasets:
        raise ValueError(f"{path}: contains no records")
    return datasets


def ranked_tasks(summary_path: Path, candidates: set[str], count: int) -> list[str]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"ranking summary does not exist: {summary_path}")
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    by_dataset = summary.get("by_dataset")
    if not isinstance(by_dataset, dict):
        raise ValueError(f"{summary_path}: expected a by_dataset mapping")
    scored: list[tuple[float, str]] = []
    missing: list[str] = []
    for name in sorted(candidates):
        group = by_dataset.get(name)
        try:
            score = float(group["methods"]["direct"]["accuracy"])
        except (KeyError, TypeError, ValueError):
            missing.append(name)
            continue
        scored.append((score, name))
    if len(scored) < count:
        raise ValueError(
            f"{summary_path}: only {len(scored)} candidate tasks have Direct accuracy; "
            f"need {count}. Missing: {', '.join(missing)}"
        )
    return [name for _, name in sorted(scored, key=lambda item: (item[0], item[1]))[:count]]


def main() -> int:
    args = parse_args()
    input_paths = [args.input.resolve(), *(path.resolve() for path in args.extra_input)]
    candidates = set().union(*(load_candidate_datasets(path) for path in input_paths))
    if args.ranking_summary:
        selected = ranked_tasks(args.ranking_summary.resolve(), candidates, args.num_tasks)
        selection = {
            "mode": "lowest_direct_accuracy",
            "ranking_summary": str(args.ranking_summary.resolve()),
            "num_tasks": args.num_tasks,
        }
    else:
        selected = list(dict.fromkeys(args.task))
        unknown = sorted(set(selected) - candidates)
        if unknown:
            raise ValueError(f"--task values absent from inputs: {', '.join(unknown)}")
        selection = {"mode": "explicit", "num_tasks": len(selected)}

    output = args.output.resolve()
    effective_eval_mode = "reasoning" if args.reasoning == "yes" else args.eval_mode
    manifest = {
        "inputs": [str(path) for path in input_paths],
        "selected_tasks": selected,
        "samples_per_task": args.samples_per_task,
        "teacher_tokens": args.teacher_tokens,
        "soft_tokens": args.soft_tokens,
        "train_batch_size": args.train_batch_size,
        "model": args.model,
        "reasoning": args.reasoning,
        "seed": args.seed,
        "unseen_evaluation": {
            "enabled": not args.skip_unseen_eval,
            "inputs": [str(path.resolve()) for path in (args.eval_input or input_paths)]
            if not args.skip_unseen_eval else [],
            "task_source": "held-out pairs from selected training tasks",
            "pairs_per_task": args.eval_pairs_per_task if not args.skip_unseen_eval else 0,
            "kind": args.periodic_eval_kind if not args.skip_unseen_eval else None,
            "mode": effective_eval_mode,
            "max_new_tokens": args.eval_max_new_tokens or (4096 if effective_eval_mode == "reasoning" else 32),
            "every_steps": args.eval_every if not args.skip_unseen_eval else 0,
            "start_step": args.eval_start_step,
            "patience_steps": args.eval_patience if not args.skip_unseen_eval else 0,
            "batch_size": args.eval_batch_size,
        },
        "save_final": args.save_final,
        "torch_runtime": str(args.torch_runtime) if args.torch_runtime else None,
        **selection,
    }
    command = [
        sys.executable, "-u", str(ROOT / "latent_remedy/train_universal_soft_suffix.py"),
        "--model", args.model, "--output", str(output),
        "--soft-tokens", str(args.soft_tokens), "--teacher-tokens", str(args.teacher_tokens),
        "--train-batch-size", str(args.train_batch_size),
        "--train-records-per-dataset", str(args.samples_per_task), "--steps", str(args.steps),
        "--learning-rate", str(args.learning_rate), "--seed", str(args.seed),
        "--eval-directions", str(args.eval_directions),
        "--generate-directions", str(args.generate_directions),
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    for input_path in input_paths:
        command.extend(("--input", str(input_path)))
    if args.reasoning == "yes":
        command.append("--explicit-reasoning")
    elif args.reasoning == "mixed":
        command.extend(("--training-prompt-modes", "no-reasoning", "reasoning"))
    if args.skip_unseen_eval:
        command.extend(("--unseen-eval-every", "0"))
    else:
        command.extend((
            "--unseen-eval-every", str(args.eval_every),
            "--unseen-eval-start-step", str(args.eval_start_step),
            "--unseen-eval-patience", str(args.eval_patience),
            "--unseen-eval-batch-size", str(args.eval_batch_size),
            "--eval-pairs-per-task", str(args.eval_pairs_per_task),
            "--periodic-eval-kind", args.periodic_eval_kind,
            "--unseen-eval-max-new-tokens",
            str(args.eval_max_new_tokens or (4096 if effective_eval_mode == "reasoning" else 32)),
        ))
        command.extend(("--unseen-eval-mode", effective_eval_mode))
    for task in selected:
        command.extend(("--include-dataset", task))
    if args.allow_download:
        command.append("--allow-download")
    if args.save_final:
        command.append("--save-final")

    evaluation_command: list[str] | None = None
    if not args.skip_unseen_eval:
        evaluation_output = output / f"unseen_eval_{effective_eval_mode}"
        modelscope_cache = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache/modelscope"))
        model_path = modelscope_cache / f"models/Qwen--Qwen3-{args.model.upper()}" / "snapshots/master"
        evaluation_command = [
            sys.executable, "-u", str(ROOT / "latent_remedy/eval_universal_soft_suffix.py"),
            "--checkpoint", str(output / "best_latent_tokens.pt"), "--model-path",
            str(model_path),
            "--output", str(evaluation_output), "--eval-mode", effective_eval_mode,
            "--max-new-tokens", str(args.eval_max_new_tokens or (4096 if effective_eval_mode == "reasoning" else 32)),
        ]
        for evaluation_input in args.eval_input or input_paths:
            evaluation_command.extend(("--input", str(evaluation_input.resolve())))
        for task in selected:
            evaluation_command.extend(("--exclude-dataset", task))
    print(json.dumps(
        {"selection": manifest, "train_command": command, "unseen_eval_command": evaluation_command},
        ensure_ascii=False, indent=2,
    ), flush=True)
    if args.dry_run:
        return 0
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise ValueError(f"output directory is not empty: {output}; pass --overwrite to replace it")
        shutil.rmtree(output)
    environment = os.environ.copy()
    if args.torch_runtime is not None:
        if not args.torch_runtime.is_dir():
            raise FileNotFoundError(f"torch runtime directory not found: {args.torch_runtime}")
        python_paths = [str(args.torch_runtime), str(ROOT / "third_party/RelayCaching")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = ":".join(python_paths)
        print(f"[setup] using torch runtime: {args.torch_runtime}", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if result.returncode != 0:
        return result.returncode
    (output / "latent_training_selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if evaluation_command is not None:
        return subprocess.run(evaluation_command, cwd=ROOT, env=environment, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
