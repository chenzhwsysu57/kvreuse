"""Step 1 of adaptive self-play: LLM-selected distributions → verified tasks.

The LLM may choose only a distribution over the fixed, executable task DSL.
It cannot supply prompt text, shared data, gold labels, or arbitrary code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from .synthetic_tasks import LAYOUTS, TASK_TYPES, generate_tasks, validate_generated_record


class JsonChat(Protocol):
    def chat_json(self, messages: list[dict[str, str]], *, temperature: float | None = None,
                  max_tokens: int | None = None, try_native_json: bool = False,
                  trace_name: str = "llm") -> Any: ...


@dataclass(frozen=True)
class AttackDistribution:
    """A validated, reproducible request to generate a batch of task pairs."""

    task_weights: dict[str, float]
    pairs: int
    min_rows: int = 4
    max_rows: int = 12
    layouts: tuple[str, ...] = LAYOUTS
    min_pairs_per_type: int = 0
    reasoning: str = ""
    rationale: str = ""

    def validate(self) -> None:
        if not isinstance(self.pairs, int) or not 1 <= self.pairs <= 2000:
            raise ValueError("pairs must be an integer from 1 to 2000")
        if not isinstance(self.min_pairs_per_type, int) or self.min_pairs_per_type < 0:
            raise ValueError("min_pairs_per_type must be a non-negative integer")
        if not self.task_weights or set(self.task_weights) - TASK_TYPES.keys():
            raise ValueError("task_weights must contain only known non-empty task types")
        if any(not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0
               for weight in self.task_weights.values()):
            raise ValueError("each task weight must be a positive number")
        if not (4 <= self.min_rows <= self.max_rows <= 100):
            raise ValueError("row bounds must satisfy 4 <= min_rows <= max_rows <= 100")
        if not self.layouts or any(layout not in LAYOUTS for layout in self.layouts):
            raise ValueError(f"layouts must be chosen from {LAYOUTS}")
        if self.min_pairs_per_type * len(self.task_weights) > self.pairs:
            raise ValueError("minimum per type exceeds pairs")
        for name, text in (("reasoning", self.reasoning), ("rationale", self.rationale)):
            if not isinstance(text, str) or len(text) > 2000:
                raise ValueError(f"{name} must be a string of at most 2000 characters")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttackDistribution":
        allowed = {"task_weights", "pairs", "min_rows", "max_rows", "layouts",
                   "min_pairs_per_type", "reasoning", "rationale"}
        extra = set(data) - allowed
        missing = {"task_weights", "pairs"} - set(data)
        if extra or missing:
            raise ValueError(f"invalid distribution fields; missing={sorted(missing)}, extra={sorted(extra)}")
        value = cls(
            task_weights=dict(data["task_weights"]), pairs=data["pairs"],
            min_rows=data.get("min_rows", 4), max_rows=data.get("max_rows", 12),
            layouts=tuple(data.get("layouts", LAYOUTS)),
            min_pairs_per_type=data.get("min_pairs_per_type", 0),
            reasoning=data.get("reasoning", ""),
            rationale=data.get("rationale", ""),
        )
        value.validate()
        return value


def allocate_counts(distribution: AttackDistribution) -> dict[str, int]:
    """Largest-remainder allocation with an optional exploration floor."""
    distribution.validate()
    names = sorted(distribution.task_weights)
    counts = {name: distribution.min_pairs_per_type for name in names}
    remaining = distribution.pairs - sum(counts.values())
    total = sum(distribution.task_weights.values())
    quotas = {name: remaining * distribution.task_weights[name] / total for name in names}
    for name in names:
        counts[name] += int(quotas[name])
    leftovers = remaining - sum(int(quota) for quota in quotas.values())
    for name in sorted(names, key=lambda item: (-(quotas[item] % 1), item))[:leftovers]:
        counts[name] += 1
    return counts


def generate_attack_batch(distribution: AttackDistribution, *, seed: int, round_index: int) -> list[dict[str, Any]]:
    """Generate exact counts and attach immutable provenance; no LLM labels enter data."""
    if not isinstance(seed, int) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError("seed must be an integer and round_index must be non-negative")
    counts = allocate_counts(distribution)
    request = asdict(distribution)
    request_id = hashlib.sha256(json.dumps([request, seed, round_index], sort_keys=True).encode()).hexdigest()[:16]
    records = list(generate_tasks(counts, seed=seed, min_rows=distribution.min_rows,
                                  max_rows=distribution.max_rows, layouts=distribution.layouts))
    if len(records) != distribution.pairs:
        raise AssertionError("generator did not honor allocated pair count")
    for record in records:
        metadata = dict(record["metadata"])
        metadata["attack_distribution"] = request
        metadata["attack_round"] = round_index
        metadata["attack_request_id"] = request_id
        record["metadata"] = metadata
        record["attack_distribution_id"] = request_id
        validate_generated_record(record)
    return records


def attacker_prompt(dashboard: Mapping[str, Any]) -> str:
    """New prompt for tool-like policy selection; deliberately independent of old prompts."""
    catalog = {name: {"dimension": dimension, "label": label}
               for name, (dimension, label) in TASK_TYPES.items()}
    return (
        "你是 cross-prefix KV-cache reuse 测试的攻击分布规划器。\n\n"
        "先分析仪表盘：\n"
        "1. 哪些任务类别满足 Full 准确率高、Reuse+当前 suffix 准确率低；\n"
        "2. 哪些类别已经被过度采样；\n"
        "3. 应保留多少探索性采样。\n"
        "随后只返回一个 JSON 对象。必须把简洁、基于统计证据的分析写入 JSON 的 `reasoning` 字段。\n\n"
        "你只能从给定任务目录中选择一个任务分布。不得编写题目文本、共享资料、答案、gold 标签、suffix、代码、新任务类型或模型配置。"
        "所有题目和标签均由确定性程序生成。不要仅因为 Full prefill 失败而选择某个分布。\n\n"
        f"任务目录：\n{json.dumps(catalog, ensure_ascii=False)}\n\n"
        f"当前防御仪表盘：\n{json.dumps(dict(dashboard), ensure_ascii=False)}\n\n"
        "请严格使用如下 JSON schema，禁止添加字段：\n"
        "{\n"
        '  "reasoning": "简洁的、基于证据的分析",\n'
        '  "task_weights": {"任务目录中的类别名": 1.0},\n'
        '  "pairs": 128,\n'
        '  "min_rows": 4,\n'
        '  "max_rows": 12,\n'
        '  "layouts": ["table", "json"],\n'
        '  "min_pairs_per_type": 4,\n'
        '  "rationale": "一句话说明本轮生成策略"\n'
        "}\n"
        "约束：所有权重必须为正数；只能使用任务目录中的类别名；pairs 必须在 1..2000；行数必须在 4..100；"
        "layouts 必须是 table/json/lines 的非空子集；min_pairs_per_type × 所选类别数不得超过 pairs。"
    )


class DistributionAttacker:
    """LLM policy layer; the program remains the sole task/label authority."""

    def __init__(self, llm: JsonChat):
        self.llm = llm

    def propose(self, dashboard: Mapping[str, Any]) -> AttackDistribution:
        data = self.llm.chat_json([{"role": "user", "content": attacker_prompt(dashboard)}],
                                  temperature=0.2, max_tokens=1200, try_native_json=True,
                                  trace_name="adaptive_distribution_attacker")
        if not isinstance(data, dict):
            raise ValueError("attacker must return one JSON object")
        return AttackDistribution.from_dict(data)