"""Step 2 of adaptive self-play: dashboard → suffix candidates."""

from __future__ import annotations

import json
import re
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
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", self.candidate_id) or self.candidate_id in {"full", "direct_reuse", "incumbent"}:
            raise ValueError("candidate_id must be a safe, non-reserved identifier")
        if not isinstance(self.reasoning, str) or len(self.reasoning) > 800:
            raise ValueError("reasoning must be a string of at most 800 characters")
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
        "按 failure_priority 中当前候选的 Full−Reuse 差距从大到小优先分析；结合 directions 样本量，"
        "不要把少数样本的大差距当作稳定结论。单个 Full 正确、Reuse 错误案例没有进一步的准确率差距大小之分。"
        "例如目标任务锚定、数据与指令分离、证据重新锚定。"
        "然后只返回一个 JSON 对象；每个候选必须在自己的 `reasoning` 字段中写出简洁的分析与设计理由。\n\n"
        "准确的执行结构是：当前有效的 target prefix → 在 source prefix 条件下计算的共享资料 KV → 你的 suffix → question。\n"
        "target prefix 中的任务规则必须保留；question 可能只是通用选项选择和输出要求，不包含完整任务。"
        "例如 target prefix 要求计数、source prefix 要求求和，而 question 只要求返回选项字母。"
        "不得指示忽略所有前置指令，或声称只有后续 question 定义任务；这会删除正确的目标规则。\n"
        "source prefix 通过共享块表示残留影响，不一定以可见文本出现；suffix 不能真正清空、修改或重新计算已有 KV。"
        "本地模型保持 no-reasoning；不要要求额外解释、思维链或改变既定答案输出协议。\n"
        "请使用 refinement_history 中所有历史候选的全文、分类型得分、修复和退化案例进行比较。"
        "每个 reasoning 必须区分：观测证据、尚未证实的失败原因假设、相对已有候选的变化、预期可测的修复与风险。"
        "没有历史或样本量很小时明确说明，不得编造效果或把注意力机制假设写成已证实原因。\n"
        "候选应探索不同机制而非重写同一句 context reset：回指当前 target prefix、区分资料与旧结论、"
        "按当前规则核对选项、对已有最佳候选做精简或单组件消融。"
        "这些只是探索方向，不是固定答案；避免重复已失败的候选，重新尝试时说明实质变化。\n"
        "不要为任何单个样例定制 suffix；最终文本必须在所有任务中相同。\n\n"
        f"仪表盘：\n{json.dumps(dict(dashboard), ensure_ascii=False)}\n\n"
        f"请严格按以下 schema 返回恰好 {candidate_count} 个候选，禁止添加任何其他字段。"
        "每个候选都必须有非空 suffix，且 suffix 去除首尾空白后必须含有可见英文文本；禁止只输出换行、空格或符号分隔线。"
        "每个 reasoning 最多800个字符。不要省略最后一个候选的 suffix。\n"
        "{\n  \"candidates\": [\n"
        "    {\"candidate_id\": \"简短且唯一的英文标识\", \"reasoning\": \"简洁的分析与理由\", \"suffix\": \"通用英文 suffix 文本\"}\n"
        "  ]\n}\n"
        "约束：suffix 必须为英文、非空、最多 800 个字符，且不得提及任何 benchmark 或任务类别名称。"
        "candidate_id 必须以英文字母开头，只包含英文字母、数字、下划线或连字符，最多64字符；"
        "不得使用 full、direct_reuse、incumbent 这些保留名称。"
    )


class SuffixDefender:
    def __init__(self, llm: JsonChat):
        self.llm = llm

    def propose(self, dashboard: Mapping[str, Any], *, candidate_count: int = 4) -> list[SuffixCandidate]:
        if not 1 <= candidate_count <= 8:
            raise ValueError("candidate_count must be 1..8")
        prompt = defender_prompt(dashboard, candidate_count=candidate_count)
        last_error = None
        for attempt in range(1, 4):
            retry = "" if attempt == 1 else (
                "\n\nYour previous candidate JSON was rejected: " + str(last_error) + ". "
                "Return a replacement JSON now. Every suffix must contain visible non-whitespace English text; "
                "whitespace-only separators are invalid."
            )
            try:
                data = self.llm.chat_json([{"role": "user", "content": prompt + retry}],
                                          temperature=0.5, max_tokens=900 * candidate_count, try_native_json=True,
                                          trace_name=f"adaptive_suffix_defender_attempt_{attempt}")
                candidates = parse_candidates(data, max_candidates=candidate_count)
                if len(candidates) != candidate_count:
                    raise ValueError(f"defender must return exactly {candidate_count} candidates")
                return candidates
            except (TypeError, ValueError) as error:
                last_error = error
        raise ValueError(f"defender returned invalid candidates after 3 attempts: {last_error}")


def candidate_manifest(candidates: list[SuffixCandidate]) -> dict[str, Any]:
    return {"candidates": [asdict(candidate) for candidate in candidates]}