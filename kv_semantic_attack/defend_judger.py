import json
from .docs_prompt import judger_prompt
from .llm_client import ChatLLM
from .schemas import AttackCase, DefenseJudgment, ExecutionResult

class DefendJudger:
    def __init__(self, llm: ChatLLM):
        self.llm = llm

    def judge(self, case: AttackCase, remedy: str,
              without: dict[str, ExecutionResult],
              with_remedy: dict[str, ExecutionResult]) -> DefenseJudgment:
        payload = {"prefix_a": case.prefix_a, "prefix_b": case.prefix_b,
                   "shared_block": case.shared_block, "remedy": remedy,
                   "question": case.question,
                   "without_remedy": {k: v.text for k, v in without.items()},
                   "with_remedy": {k: v.text for k, v in with_remedy.items()}}
        raw = self.llm.chat([{"role": "user", "content": judger_prompt(
            "defend judger.txt", json.dumps(payload, ensure_ascii=False, indent=2)
        )}], temperature=0.0, max_tokens=700,
            trace_name="defend_judger")
        data = self.llm._extract_json(raw)
        if not isinstance(data, dict) or "summary" not in data or "Defense Success" not in data:
            raise ValueError("Defend Judger returned invalid JSON")
        summary = str(data["summary"]) + (
            f"\n长度反馈：本次 remedy 长度为 {len(remedy)}。后续请在保持防御效果的"
            "前提下继续尝试更短的 remedy。"
        )
        return DefenseJudgment(summary=summary,
                               defense_success=bool(data["Defense Success"]),
                               raw_output=raw, details=data)
