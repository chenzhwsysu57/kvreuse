"""CLI for the real five-round semantic self-play experiment."""

import argparse
import os
from pathlib import Path

from kv_semantic_attack import (
    Attacker,
    AttackJudger,
    ChatLLM,
    Defender,
    DefendJudger,
    LLMConfig,
    SelfPlayOrchestrator,
)
from kv_semantic_attack.mock_executor import MockExecutor
from kv_semantic_attack.step_logger import StepLogger


def load_local_env(path: Path = Path(".env.local")) -> None:
    """Load simple KEY=VALUE entries without adding a dotenv dependency."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=("0.6b", "1.7b", "4b", "8b"), default="0.6b")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reasoning", dest="reasoning", action="store_true")
    mode.add_argument("--no-reasoning", dest="reasoning", action="store_false")
    parser.set_defaults(reasoning=True)
    parser.add_argument("--log-dir", default="./selfplay_logs")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--mock", action="store_true", help="run orchestration without Qwen3")
    parser.add_argument("--debug", action="store_true", help="print every LLM/executor input and output")
    parser.add_argument("--attacks-per-round", type=int, default=1)
    parser.add_argument("--defenses-per-round", type=int, default=1)
    parser.add_argument("--attacker-workers", type=int, default=4,
                        help="concurrent attacker API calls per round")
    args = parser.parse_args()

    load_local_env()
    step_logger = StepLogger(args.log_dir)
    llm_cfg = LLMConfig(
        base_url=os.getenv("DASHSCOPE_BASE_URL", ""),
        model=os.getenv("DASHSCOPE_MODEL", "qwen3.8-max"),
        timeout=60.0,
        max_retries=1,
        enable_thinking=False,
    )
    if not llm_cfg.base_url:
        raise RuntimeError("DASHSCOPE_BASE_URL is missing from .env.local")
    llm = ChatLLM(
        llm_cfg,
        trace_dir=None,
        debug=args.debug,
        step_logger=step_logger,
    )
    if args.mock:
        executor = MockExecutor(step_logger=step_logger)
    else:
        from kv_semantic_attack.qwen_executor import Qwen3KVReuseExecutor
        executor = Qwen3KVReuseExecutor(
            model_size=args.model_size,
            reasoning=args.reasoning,
            trace_dir=None,
            allow_download=args.allow_download,
            debug=args.debug,
            step_logger=step_logger,
        )
    runner = SelfPlayOrchestrator(
        attacker=Attacker(llm),
        attack_judger=AttackJudger(llm),
        defender=Defender(llm),
        defend_judger=DefendJudger(llm),
        executor=executor,
        max_rounds=5, max_attempts=3,
        attacks_per_round=args.attacks_per_round,
        defenses_per_round=args.defenses_per_round,
        attacker_workers=args.attacker_workers,
        log_dir=args.log_dir,
    )
    final_remedy = runner.run(initial_remedy="")
    print("Final REMEDY:")
    print(final_remedy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
