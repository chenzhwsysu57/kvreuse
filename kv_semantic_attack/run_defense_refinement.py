#!/usr/bin/env python3
"""Run five fixed-pool defense rounds with cumulative candidate feedback."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
sys.path.insert(0, str(ROOT))

from kv_semantic_attack.build_adaptive_dashboard import build_dashboard
from kv_semantic_attack.defense_refinement import candidate_feedback, refinement_summary
from kv_semantic_attack.run_journal import file_ref, write_step


def run(command: list[str]) -> None:
    print("[refinement] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="fixed attack-pool JSONL")
    parser.add_argument("--dashboard", type=Path, required=True, help="initial dashboard with failure examples")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--stop-on-threshold", action="store_true",
                        help="optional early stop; by default complete every round")
    parser.add_argument("--candidates-per-attempt", type=int, default=4)
    parser.add_argument("--minimum-improvement-pp", type=float, default=3.0)
    parser.add_argument("--target-accuracy", type=float,
                        help="stop after a round whose best observed candidate reaches this absolute accuracy")
    parser.add_argument("--model", choices=("0.6b", "1.7b", "4b", "8b"), default="1.7b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gpu-slots", nargs="+", metavar="GPU=SLOTS",
                        help="parallel evaluation, e.g. 1=2 2=2 3=2; omit for serial")
    parser.add_argument("--max-used-gib", type=float, default=4.0)
    args = parser.parse_args()
    if args.gpu_slots:
        from kv_semantic_attack.run_multi_gpu_defense import parse_gpu_slots
        parse_gpu_slots(args.gpu_slots)
    if args.max_used_gib < 0 or args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("invalid memory, batch or token limit")
    if args.start_step < 1 or args.max_attempts < 1 or not 1 <= args.candidates_per_attempt <= 7:
        parser.error("invalid step, attempt count, or candidate count")
    if args.minimum_improvement_pp < 0:
        parser.error("minimum improvement must be non-negative")
    if args.target_accuracy is not None and not 0 <= args.target_accuracy <= 1:
        parser.error("target accuracy must be between zero and one")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("run-dir must be new or empty; do not overwrite prior experiments")
    if args.input.resolve() == args.dashboard.resolve():
        parser.error("--input and --dashboard must differ")
    dashboard = json.loads(args.dashboard.read_text(encoding="utf-8"))
    sources = {row["task_id"]: row for row in (json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines())}
    incumbent = str(dashboard.get("current_suffix", ""))
    attempts = []
    history = list(dashboard.get("refinement_history", []))
    initial_accuracy = None
    threshold_reached = False
    for attempt in range(1, args.max_attempts + 1):
        step = args.start_step + 3 * (attempt - 1)
        suffixes = args.run_dir / f"refine_{attempt:02d}_suffixes.json"
        evaluated_suffixes = args.run_dir / f"refine_{attempt:02d}_evaluated_suffixes.json"
        evaluation = args.run_dir / f"refine_{attempt:02d}_evaluation.json"
        next_dashboard = args.run_dir / f"refine_{attempt:02d}_dashboard.json"
        # Include incumbent as a candidate so selection is always a direct test,
        # never an inference from a previous random seed or batch ordering.
        dashboard["current_suffix"] = incumbent
        dashboard["refinement_history"] = history
        dashboard["search_round"] = {"index": attempt, "total": args.max_attempts,
                                     "fixed_pool": file_ref(args.input)}
        dashboard_path = args.run_dir / f"refine_{attempt:02d}_input_dashboard.json"
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run([str(PYTHON), "kv_semantic_attack/run_adaptive_defender.py", "--dashboard", str(dashboard_path),
             "--candidates", str(args.candidates_per_attempt), "--output", str(suffixes),
             "--run-dir", str(args.run_dir), "--step", str(step)])
        candidate_data = json.loads(suffixes.read_text(encoding="utf-8"))
        if incumbent.strip() and not any(c["suffix"].strip() == incumbent.strip() for c in candidate_data["candidates"]):
            name = "previous_best"
            while name in {c["candidate_id"] for c in candidate_data["candidates"]}:
                name += "_x"
            candidate_data["candidates"].append({"candidate_id": name, "reasoning": "Best previously observed suffix.", "suffix": incumbent})
        # Do not mutate the API candidate file after its step manifest was hashed.
        evaluated_suffixes.write_text(json.dumps(candidate_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        common = ["--input", str(args.input), "--candidates", str(evaluated_suffixes),
                  "--model", args.model, "--batch-size", str(args.batch_size),
                  "--max-new-tokens", str(args.max_new_tokens), "--output", str(evaluation),
                  "--run-dir", str(args.run_dir), "--step", str(step + 1)]
        if args.gpu_slots:
            run([str(PYTHON), "kv_semantic_attack/run_multi_gpu_defense.py", *common,
                 "--output-dir", str(args.run_dir / f"refine_{attempt:02d}_parallel"),
                 "--max-used-gib", str(args.max_used_gib), "--gpu-slots", *args.gpu_slots])
        else:
            run([str(PYTHON), "kv_semantic_attack/evaluate_suffix_candidates.py", *common,
                 "--reuse-engine", "scatter"])
        report = json.loads(evaluation.read_text(encoding="utf-8"))
        summary = refinement_summary(report, incumbent_suffix=incumbent,
                                    minimum_improvement_pp=args.minimum_improvement_pp)
        if initial_accuracy is None:
            initial_accuracy = summary["incumbent"]["accuracy"]
        selected = summary["best_observed"]
        improvement = 100 * (selected["accuracy"] - initial_accuracy)
        threshold_reached = improvement >= args.minimum_improvement_pp and improvement > 0
        summary["improvement_from_initial_pp"] = improvement
        summary["threshold_reached_from_initial"] = threshold_reached
        summary["target_accuracy"] = args.target_accuracy
        summary["target_reached"] = args.target_accuracy is not None and selected["accuracy"] >= args.target_accuracy
        history.append({"round": attempt, "selection": summary,
                        "candidate_feedback": candidate_feedback(report, sources, incumbent_suffix=incumbent)})
        write_step(args.run_dir, step + 2, "refinement_selection", {
            "attempt": attempt, "input_dashboard": file_ref(dashboard_path), "suffixes": file_ref(suffixes),
            "evaluated_suffixes": file_ref(evaluated_suffixes),
            "evaluation": file_ref(evaluation), "selection": summary,
        })
        attempts.append(summary)
        # Keep even small improvements for search; the threshold is a separate goal.
        incumbent = selected["suffix"]
        dashboard = build_dashboard(report, sources, candidate_id=selected["candidate_id"], examples_per_type=6)
        dashboard["current_suffix"] = incumbent
        dashboard["refinement_history"] = history
        next_dashboard.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if summary["target_reached"] or (threshold_reached and args.stop_on_threshold):
            break
    target_reached = bool(attempts and attempts[-1]["target_reached"])
    result = {"accepted": threshold_reached or target_reached, "suffix": incumbent, "attempts": attempts,
              "rounds_completed": len(attempts), "history": history,
              "target_accuracy": args.target_accuracy, "target_reached": target_reached,
              "note": "Fixed-pool search result; not a held-out generalization estimate."}
    (args.run_dir / "refinement_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status_path = "accepted_suffix.json" if threshold_reached else "refinement_stopped.json"
    (args.run_dir / status_path).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())