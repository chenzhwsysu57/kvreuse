import json
from .docs_prompt import judger_prompt
from .llm_client import ChatLLM
from .schemas import AttackCase, AttackJudgment, ExecutionResult

class AttackJudger:
    def __init__(self, llm: ChatLLM):
        self.llm = llm

    def judge(self, case: AttackCase, executions: dict[str, ExecutionResult]) -> AttackJudgment:
        payload = {"prefix_a": case.prefix_a, "prefix_b": case.prefix_b,
                   "shared_block": case.shared_block, "question": case.question,
                   "executions": {k: v.text for k, v in executions.items()}}
        raw = self.llm.chat([{"role": "user", "content": judger_prompt(
            "attack judger.txt", json.dumps(payload, ensure_ascii=False, indent=2)
        )}], temperature=0.0, max_tokens=700,
            trace_name="attack_judger")
        data = self.llm._extract_json(raw)
        if not isinstance(data, dict) or "summary" not in data or "Attack Success" not in data:
            raise ValueError("Attack Judger returned invalid JSON")
        return AttackJudgment(summary=str(data["summary"]),
                              attack_success=bool(data["Attack Success"]),
                              raw_output=raw, details=data)
