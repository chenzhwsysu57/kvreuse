import re
import uuid

from .docs_prompt import attacker_prompt
from .llm_client import ChatLLM
from .schemas import AttackCase


class Attacker:
    def __init__(self, llm: ChatLLM):
        self.llm = llm

    def propose_one(self, *, attack_history: list[str],
                    defense_history: list[str]) -> AttackCase:
        context = attacker_prompt(attack_history, defense_history)
        data = self.llm.chat_json([
            {"role": "user", "content": context},
        ], trace_name="attacker")
        if not isinstance(data, dict):
            raise ValueError("Attacker output must be a JSON object")
        allowed = {
            "reasoning", "prefix_a", "prefix_b", "shared_block", "question",
            "gold_a", "gold_b",
        }
        unexpected = sorted(set(data) - allowed)
        if unexpected:
            raise ValueError(f"Attacker added unsupported fields: {unexpected}")
        required = (
            "reasoning", "prefix_a", "prefix_b", "shared_block", "question",
            "gold_a", "gold_b",
        )
        missing = [key for key in required if not str(data.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Attacker omitted fields: {missing}")
        for key in ("gold_a", "gold_b"):
            if not re.fullmatch(r"[A-H]", str(data[key]).strip(), re.I):
                raise ValueError(f"{key} must be one plain letter A-H")
            data[key] = str(data[key]).strip().upper()
        if data["gold_a"] == data["gold_b"]:
            raise ValueError("gold_a and gold_b must be different")
        return AttackCase(
            prefix_a=str(data["prefix_a"]).strip(), prefix_b=str(data["prefix_b"]).strip(),
            shared_block=str(data["shared_block"]).strip(), question=str(data["question"]).strip(),
            gold_a=str(data["gold_a"]).strip(), gold_b=str(data["gold_b"]).strip(),
            case_id=uuid.uuid4().hex[:8],
            reasoning=str(data["reasoning"]).strip(),
        )
