"""Step 2 of adaptive self-play: dashboard → suffix candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol


class JsonChat(Protocol):
    def chat_json(self, messages: list[dict[str, str]], *, temperature: float | None = None,
                  max_tokens: int | None = None, try_native_json: bool = False,
                  trace_name: str = "llm") -> Any: ...


@dataclass(frozen=True)
class SuffixCandidate:
    candidate_id: str
    reasoning: str
    suffix: str

    def validate(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id or len(self.candidate_id) > 64:
            raise ValueError("candidate_id must be a non-empty string of at most 64 characters")
        if not isinstance(self.reasoning, str) or len(self.reasoning) > 2000:
            raise ValueError("reasoning must be a string of at most 2000 characters")
        if not isinstance(self.suffix, str) or not self.suffix.strip() or len(self.suffix) > 800:
            raise ValueError("suffix must be non-empty and at most 800 characters")
        if "\x00" in self.suffix:
            raise ValueError("suffix must not contain null bytes")


def parse_candidates(data: Any, *, max_candidates: int = 8) -> list[SuffixCandidate]:
    if not isinstance(data, dict) or set(data) != {"candidates"} or not isinstance(data["candidates"], list):
        raise ValueError("defender must return exactly {'candidates': [...]} JSON")
    if not 1 <= len(data["candidates"]) <= max_candidates:
        raise ValueError(f"defender must return 1..{max_candidates} candidates")
    candidates = []
    for item in data["candidates"]:
        if not isinstance(item, dict) or set(item) != {"candidate_id", "reasoning", "suffix"}:
            raise ValueError("each candidate needs exactly candidate_id, reasoning, suffix")
        candidate = SuffixCandidate(**item)
        candidate.validate()
        candidates.append(candidate)
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate_id values must be unique")
    if len({candidate.suffix.strip() for candidate in candidates}) != len(candidates):
        raise ValueError("candidate suffixes must be unique")
    return candidates


def defender_prompt(dashboard: Mapping[str, Any], *, candidate_count: int) -> str:
    return (
        "你负责为 cross-prefix KV-cache reuse 设计简短、通用的后置 suffix。"
        "suffix 被插在复用的共享资料 KV 块之后、最终问题之前。它必须对所有任务完全相同："
        "不得包含任务特定资料、答案、选项、模型能力声明或新的具体任务指令。\n\n"
        "先分析仪表盘：找出 Full 高但 Reuse 低的语义冲突类别。随后提出机制不同的候选，"
        "例如目标任务锚定、数据与指令分离、证据重新锚定。"
        "然后只返回一个 JSON 对象；每个候选必须在自己的 `reasoning` 字段中写出简洁的分析与设计理由。\n\n"
        "以下是软组件清单，不是要求照抄的固定模板。候选应按失败样例选择、组合、简化或改写这些作用：\n"
        "- 明确后续问题定义当前唯一任务；\n"
        "- 将前面的共享内容视为当前任务可用的数据／证据；\n"
        "- 不继承共享内容可能残留的旧目标、规则或输出格式；\n"
        "- 要求根据后续问题重新解释相关资料。\n"
        "不要为任何单个样例定制 suffix；最终文本必须在所有任务中相同。\n\n"
        f"仪表盘：\n{json.dumps(dict(dashboard), ensure_ascii=False)}\n\n"
        f"请严格按以下 schema 返回恰好 {candidate_count} 个候选，禁止添加任何其他字段：\n"
        "{\n  \"candidates\": [\n"
        "    {\"candidate_id\": \"简短且唯一的英文标识\", \"reasoning\": \"简洁的分析与理由\", \"suffix\": \"通用英文 suffix 文本\"}\n"
        "  ]\n}\n"
        "约束：suffix 必须为英文、非空、最多 800 个字符，且不得提及任何 benchmark 或任务类别名称。"
    )


class SuffixDefender:
    def __init__(self, llm: JsonChat):
        self.llm = llm

    def propose(self, dashboard: Mapping[str, Any], *, candidate_count: int = 4) -> list[SuffixCandidate]:
        if not 1 <= candidate_count <= 8:
            raise ValueError("candidate_count must be 1..8")
        data = self.llm.chat_json([{"role": "user", "content": defender_prompt(dashboard, candidate_count=candidate_count)}],
                                  temperature=0.5, max_tokens=2400, try_native_json=True,
                                  trace_name="adaptive_suffix_defender")
        return parse_candidates(data, max_candidates=candidate_count)


def candidate_manifest(candidates: list[SuffixCandidate]) -> dict[str, Any]:
    return {"candidates": [asdict(candidate) for candidate in candidates]}