#!/usr/bin/env python3
"""Call Qwen 3.8 Max once to propose universal suffix candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kv_semantic_attack.adaptive_defense import SuffixDefender, candidate_manifest
from kv_semantic_attack.config import LLMConfig
from kv_semantic_attack.llm_client import ChatLLM
from kv_semantic_attack.run_selfplay import load_local_env
from kv_semantic_attack.run_journal import file_ref, write_step
from kv_semantic_attack.step_logger import StepLogger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "kv_semantic_attack/adaptive_runs/logs")
    parser.add_argument("--run-dir", type=Path, help="write immutable per-step manifest and API traces here")
    parser.add_argument("--step", type=int, help="positive environment step number; required with --run-dir")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if (args.run_dir is None) != (args.step is None):
        parser.error("--run-dir and --step must be provided together")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    dashboard = json.loads(args.dashboard.read_text(encoding="utf-8"))
    load_local_env()
    config = LLMConfig(base_url=os.getenv("DASHSCOPE_BASE_URL", ""),
                       model=os.getenv("DASHSCOPE_MODEL", "qwen3.8-max"),
                       timeout=60.0, max_retries=1, enable_thinking=False)
    if not config.base_url:
        raise RuntimeError("DASHSCOPE_BASE_URL is missing from .env.local")
    log_dir = args.run_dir / "api_traces" if args.run_dir else args.log_dir
    defender = SuffixDefender(ChatLLM(config, debug=args.debug, step_logger=StepLogger(str(log_dir))))
    candidates = defender.propose(dashboard, candidate_count=args.candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidate_manifest(candidates), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.run_dir:
        manifest = write_step(args.run_dir, args.step, "defense_candidates", {
            "api_trace_dir": str(log_dir.resolve()), "dashboard": file_ref(args.dashboard),
            "candidate_count": args.candidates, "suffix_candidates": candidate_manifest(candidates),
            "candidate_file": file_ref(args.output),
        })
        print(f"step_manifest: {manifest}")
    print(json.dumps(candidate_manifest(candidates), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())