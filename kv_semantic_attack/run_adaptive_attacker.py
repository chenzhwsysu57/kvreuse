#!/usr/bin/env python3
"""Run Qwen 3.8 Max once: dashboard → attack distribution → generated batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kv_semantic_attack.adaptive_attack import DistributionAttacker, allocate_counts, generate_attack_batch
from kv_semantic_attack.config import LLMConfig
from kv_semantic_attack.llm_client import ChatLLM
from kv_semantic_attack.run_selfplay import load_local_env
from kv_semantic_attack.run_journal import file_ref, write_step
from kv_semantic_attack.step_logger import StepLogger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, required=True,
                        help="JSON summary with current_suffix, per-type metrics and coverage")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--pairs", type=int,
                        help="force total A/B task pairs after the LLM selects its distribution (1..2000)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "kv_semantic_attack/adaptive_runs/logs")
    parser.add_argument("--run-dir", type=Path, help="write immutable per-step manifest and API traces here")
    parser.add_argument("--step", type=int, help="positive environment step number; required with --run-dir")
    parser.add_argument("--debug", action="store_true", help="print the exact LLM request and response")
    args = parser.parse_args()
    if (args.run_dir is None) != (args.step is None):
        parser.error("--run-dir and --step must be provided together")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    dashboard = json.loads(args.dashboard.read_text(encoding="utf-8"))
    if not isinstance(dashboard, dict):
        parser.error("dashboard must be a JSON object")
    load_local_env()
    config = LLMConfig(base_url=os.getenv("DASHSCOPE_BASE_URL", ""),
                       model=os.getenv("DASHSCOPE_MODEL", "qwen3.8-max"),
                       timeout=60.0, max_retries=1, enable_thinking=False)
    if not config.base_url:
        raise RuntimeError("DASHSCOPE_BASE_URL is missing from .env.local")
    log_dir = args.run_dir / "api_traces" if args.run_dir else args.log_dir
    logger = StepLogger(str(log_dir))
    attacker = DistributionAttacker(ChatLLM(config, debug=args.debug, step_logger=logger))
    proposal = attacker.propose(dashboard)
    if args.pairs is not None:
        proposal = replace(proposal, pairs=args.pairs)
        proposal.validate()
    records = generate_attack_batch(proposal, seed=args.seed, round_index=args.round)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                           encoding="utf-8")
    if args.run_dir:
        manifest = write_step(args.run_dir, args.step, "attack_distribution", {
            "api_trace_dir": str(log_dir.resolve()), "dashboard": file_ref(args.dashboard),
            "proposal": proposal.__dict__, "allocated_counts": allocate_counts(proposal),
            "pairs_override": args.pairs, "seed": args.seed, "round": args.round,
            "generated_candidates": file_ref(args.output),
        })
        print(f"step_manifest: {manifest}")
    print(json.dumps({"proposal": proposal.__dict__, "counts": allocate_counts(proposal),
                      "pairs": len(records), "output": str(args.output), "log_dir": str(args.log_dir)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())