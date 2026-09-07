#!/usr/bin/env python3
"""Iteratively refine suffixes on one fixed attack pool until threshold success."""

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
from kv_semantic_attack.defense_refinement import refinement_summary
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
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--candidates-per-attempt", type=int, default=4)
    parser.add_argument("--minimum-improvement-pp", type=float, default=3.0)
    parser.add_argument("--model", choices=("0.6b", "1.7b", "4b", "8b"), default="1.7b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.start_step < 1 or args.max_attempts < 1 or not 1 <= args.candidates_per_attempt <= 8:
        parser.error("invalid step, attempt count, or candidate count")
    if args.input.resolve() == args.dashboard.resolve():
        parser.error("--input and --dashboard must differ")
    dashboard = json.loads(args.dashboard.read_text(encoding="utf-8"))
    sources = {row["task_id"]: row for row in (json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines())}
    incumbent = str(dashboard.get("current_suffix", ""))
    attempts = []
    for attempt in range(1, args.max_attempts + 1):
        step = args.start_step + 3 * (attempt - 1)
        suffixes = args.run_dir / f"refine_{attempt:02d}_suffixes.json"
        evaluation = args.run_dir / f"refine_{attempt:02d}_evaluation.json"
        next_dashboard = args.run_dir / f"refine_{attempt:02d}_dashboard.json"
        # Include incumbent as a candidate so selection is always a direct test,
        # never an inference from a previous random seed or batch ordering.
        if incumbent.strip():
            dashboard["current_suffix"] = incumbent
        dashboard_path = args.run_dir / f"refine_{attempt:02d}_input_dashboard.json"
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run([str(PYTHON), "kv_semantic_attack/run_adaptive_defender.py", "--dashboard", str(dashboard_path),
             "--candidates", str(args.candidates_per_attempt), "--output", str(suffixes),
             "--run-dir", str(args.run_dir), "--step", str(step)])
        candidate_data = json.loads(suffixes.read_text(encoding="utf-8"))
        if incumbent.strip():
            candidate_data["candidates"].append({"candidate_id": "incumbent", "reasoning": "Previous accepted suffix.", "suffix": incumbent})
            suffixes.write_text(json.dumps(candidate_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run([str(PYTHON), "kv_semantic_attack/evaluate_suffix_candidates.py", "--input", str(args.input),
             "--candidates", str(suffixes), "--model", args.model, "--batch-size", str(args.batch_size),
             "--max-new-tokens", str(args.max_new_tokens), "--reuse-engine", "scatter", "--output", str(evaluation),
             "--run-dir", str(args.run_dir), "--step", str(step + 1)])
        report = json.loads(evaluation.read_text(encoding="utf-8"))
        summary = refinement_summary(report, incumbent_suffix=incumbent,
                                    minimum_improvement_pp=args.minimum_improvement_pp)
        write_step(args.run_dir, step + 2, "refinement_selection", {
            "attempt": attempt, "input_dashboard": file_ref(dashboard_path), "suffixes": file_ref(suffixes),
            "evaluation": file_ref(evaluation), "selection": summary,
        })
        attempts.append(summary)
        selected = summary["selected"]
        if summary["accepted"]:
            result = {"accepted": True, "attempt": attempt, "suffix": selected["suffix"], "attempts": attempts}
            (args.run_dir / "accepted_suffix.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        # No acceptance: retain incumbent, rebuild diagnostic failure examples
        # for the next defense attempt on exactly the same candidate pool.
        dashboard = build_dashboard(report, sources, candidate_id=selected["candidate_id"], examples_per_type=3)
        dashboard["current_suffix"] = incumbent
        next_dashboard.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"accepted": False, "incumbent_suffix": incumbent, "attempts": attempts}
    (args.run_dir / "refinement_stopped.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())