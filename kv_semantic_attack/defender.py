import re

from .docs_prompt import defender_prompt
from .llm_client import ChatLLM
from .schemas import DefenseProposal


class Defender:
    def __init__(self, llm: ChatLLM):
        self.llm = llm

    def propose_one(self, *, attack_history: list[str],
                    defense_history: list[str]) -> DefenseProposal:
        context = defender_prompt(attack_history, defense_history)
        data = self.llm.chat_json([
            {"role": "user", "content": context},
        ], trace_name="defender")
        if not isinstance(data, dict):
            raise ValueError("Defender output must be a JSON object")
        reasoning = str(data.get("reasoning", "")).strip()
        remedy = str(data.get("remedy", "")).strip()
        if not remedy:
            raise ValueError("Defender returned an empty remedy")
        return DefenseProposal(reasoning=reasoning, remedy=remedy)
