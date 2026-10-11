#!/usr/bin/env python3
"""Dataset adapter for evaluating RelayCaching methods on this repository's A/B tasks.

The reusable segment is represented using RelayCaching's native agent placeholder
syntax (``{agent_<side>_current}``), while the target prompt retains the project's
no-reasoning, boxed-answer contract.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SYSTEM_MESSAGE = "You're a helpful assistant."
BOXED_PREFILL = "The answer is: \\boxed{"
NO_REASONING = (
    "Do not explain or reason. Output only the final answer in exactly one \\boxed{...}."
)
EXPLICIT_REASONING = (
    "First reason about the task step by step. Then put your final answer "
    "in exactly one \\boxed{...}."
)


@dataclass(frozen=True)
class RelayExample:
    task_id: str
    dataset: str
    source: str
    target: str
    source_prefix: str
    target_prefix: str
    shared_block: str
    question: str
    gold: str

    @property
    def placeholder(self) -> str:
        return "{agent_" + self.source + "_current}"

    def source_user_content(self) -> str:
        """Prompt whose block cache is sent as the upstream agent message."""
        return f"{self.source_prefix}\n\n{self.shared_block}"

    def target_user_template(self, *, explicit_reasoning: bool = False) -> str:
        """RelayCaching-compatible target template before placeholder replacement."""
        instruction = EXPLICIT_REASONING if explicit_reasoning else NO_REASONING
        return (
            f"{self.target_prefix}\n\n{self.placeholder}\n\n{self.question}\n\n{instruction}"
        )

    def target_user_full(self, *, explicit_reasoning: bool = False) -> str:
        """Dense/full counterpart with exactly the same textual shared block."""
        return self.target_user_template(explicit_reasoning=explicit_reasoning).replace(
            self.placeholder, self.shared_block
        )

    def as_metadata(self, *, explicit_reasoning: bool = False) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "dataset": self.dataset,
            "source_side": self.source,
            "target_side": self.target,
            "gold": self.gold,
            "source_user_content": self.source_user_content(),
            "target_user_template": self.target_user_template(explicit_reasoning=explicit_reasoning),
            "target_user_full": self.target_user_full(explicit_reasoning=explicit_reasoning),
            "explicit_reasoning": explicit_reasoning,
            "assistant_prefill": None if explicit_reasoning else BOXED_PREFILL,
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_examples(records: Iterable[dict[str, Any]], *, directions: Iterable[str] = ("a_to_b", "b_to_a")) -> Iterable[RelayExample]:
    mapping = {"a_to_b": ("a", "b"), "b_to_a": ("b", "a")}
    for record in records:
        for direction in directions:
            source, target = mapping[direction]
            yield RelayExample(
                task_id=record["task_id"], dataset=record["dataset"],
                source=source, target=target,
                source_prefix=record[f"prefix_{source}"],
                target_prefix=record[f"prefix_{target}"],
                shared_block=record["shared_block"], question=record["question"],
                gold=record[f"gold_{target}"],
            )


def parse_prediction(dataset: str, text: str) -> str | None:
    boxed = re.findall(r"\\boxed\s*\{\s*([^{}\s]+)\s*\}", text, flags=re.I)
    if boxed:
        return boxed[-1].strip().upper()
    # Keep this fallback identical in spirit to the existing project parser.
    answer = re.findall(r"(?:FINAL|ANSWER|OPTION|CHOICE)\s*(?:IS|:|=)\s*\[?([A-H])\]?", text.upper())
    return answer[-1] if answer else None


def write_metadata(path: Path, records: list[dict[str, Any]], *, explicit_reasoning: bool = False) -> None:
    rows = [example.as_metadata(explicit_reasoning=explicit_reasoning) for example in iter_examples(records)]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
