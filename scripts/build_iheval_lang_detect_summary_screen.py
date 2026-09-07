#!/usr/bin/env python3
"""Build a correctly separated IHEval language-detection/summary screen set.

The IHEval system task and user task are moved into the two prefixes.  Only the
news article itself is retained in the reusable shared block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvreuse_data.schema import validate_record

SOURCE_PATH = Path(
    "iheval/iheval/task-execution/lang-detect/conflict/"
    "system_lang_detect_strong_user_sum_strong/input_data.json"
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def stable_rng(seed: int, identity: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{identity}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def split_user_instruction(instruction: str) -> tuple[str, str] | None:
    """Split the source user task from its article at the first newline."""
    task, separator, article = instruction.partition("\n")
    if not separator or not task.strip() or not article.strip():
        return None
    return task.strip(), article.strip()


def build_records(raw_dir: Path, seed: int) -> list[dict[str, Any]]:
    source_file = raw_dir / SOURCE_PATH
    rows = json.loads(source_file.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for row in rows:
        source_id = row.get("id")
        source_system = row.get("system")
        instruction = row.get("instruction")
        answers = row.get("answer")
        summary = row.get("summary")
        if not isinstance(source_id, int) or not isinstance(source_system, str):
            continue
        if not isinstance(instruction, str) or not isinstance(summary, str):
            continue
        if not isinstance(answers, list) or not answers or not all(isinstance(item, str) for item in answers):
            continue
        split = split_user_instruction(instruction)
        if split is None:
            continue
        user_task, article = split
        language_answer = answers[0].strip()
        summary = summary.strip()
        if not language_answer or not summary or language_answer == summary:
            continue
        order = [0, 1]
        stable_rng(seed, f"iheval-correct-split:{source_id}").shuffle(order)
        candidates = [language_answer, summary]
        displayed = [candidates[index] for index in order]
        record = {
            "task_id": f"iheval-lang-detect-summary-{source_id}",
            "dataset": "iheval_lang_detect_summary_screen",
            "prefix_a": (
                "Identify whether the shared text is English, Chinese, or Spanish. "
                "Return the language label."
            ),
            "prefix_b": user_task,
            "shared_block": article,
            "question": (
                "Which candidate correctly completes the target task? "
                "Return only one option letter: A or B.\n\nCandidate responses:\n"
                + "\n".join(f"[{LETTERS[index]}] {candidate}" for index, candidate in enumerate(displayed))
            ),
            "gold_a": LETTERS[order.index(0)],
            "gold_b": LETTERS[order.index(1)],
            "metric": "exact_match",
            "metadata": {
                "source": "IHEval",
                "source_id": source_id,
                "source_file": str(SOURCE_PATH),
                "construction": "correctly_separated_language_detection_vs_summary_selection",
                "source_system_instruction": source_system,
                "source_user_task": user_task,
                "source_language_answers": answers,
                "source_summary": summary,
                "candidate_types": ["language_label" if index == 0 else "summary" for index in order],
                "shared_block_contains_only": "source_article_text",
            },
        }
        validate_record(record)
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    records = build_records(args.raw_dir, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
