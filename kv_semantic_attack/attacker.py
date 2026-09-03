import json
import uuid
from typing import List, Sequence

from .llm_client import ChatLLM
from .schemas import AttackCase, VerifiedAttack


ATTACKER_SYSTEM_PROMPT = """
You are a red-team task synthesizer for a KV-reuse system.

Your job is to propose NEW synthetic task-conflict cases that may cause the
target model to fail specifically because a shared text block is reused under
a different task context.

Important constraints:
1. Do NOT propose a defense or correction.
2. Do NOT use explicit jailbreak phrases such as "ignore all future instructions".
3. The shared_block must be ordinary task content, not an attack message to the model.
4. source_instruction and target_instruction should impose meaningfully different,
   preferably conflicting, ways of using the SAME shared_block.
5. The task must have clear, objectively checkable expected outputs.
6. The same exact shared_block must be usable in both source and target conditions.
7. Prefer diverse task families instead of repeating previous attacks.
8. The final system, not you, decides whether an attack is valid.
9. Return JSON only.
""".strip()


class Attacker:
    def __init__(self, llm: ChatLLM):
        self.llm = llm

    def propose(
        self,
        *,
        correction: str,
        history: Sequence[VerifiedAttack],
        n: int,
    ) -> List[AttackCase]:
        history_view = []
        for item in history:
            if not item.accepted:
                continue
            a = item.attack
            history_view.append({
                "source_instruction": a.source_instruction,
                "target_instruction": a.target_instruction,
                "shared_block": a.shared_block,
                "query": a.query,
                "expected_target_answer": a.expected_target_answer,
                "reuse_output": item.reuse_result.text,
                "failure_strength": item.failure_strength,
                "semantic_family": a.semantic_family,
            })

        user_prompt = f"""
Current post-reuse correction:
{correction if correction else "<EMPTY>"}

Previously VERIFIED counterexamples:
{json.dumps(history_view, ensure_ascii=False, indent=2)}

Generate {n} new attack proposals that are likely to falsify the current
correction. Try to cover semantic conflict patterns not already represented.

Return a JSON array. Each item must have exactly these fields:
- source_instruction
- target_instruction
- shared_block
- query
- expected_source_answer
- expected_target_answer
- eval_type: one of ["exact", "contains"]
- eval_metadata: object
- attack_rationale
- semantic_family
""".strip()

        data = self.llm.chat_json([
            {"role": "system", "content": ATTACKER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        if isinstance(data, dict) and "attacks" in data:
            data = data["attacks"]
        if not isinstance(data, list):
            raise ValueError("Attacker output must be a JSON list.")

        attacks: List[AttackCase] = []
        for x in data[:n]:
            attacks.append(AttackCase(
                attack_id=str(uuid.uuid4())[:8],
                source_instruction=str(x["source_instruction"]),
                target_instruction=str(x["target_instruction"]),
                shared_block=str(x["shared_block"]),
                query=str(x.get("query", "")),
                expected_source_answer=str(x["expected_source_answer"]),
                expected_target_answer=str(x["expected_target_answer"]),
                eval_type=str(x.get("eval_type", "exact")),
                eval_metadata=dict(x.get("eval_metadata", {})),
                attack_rationale=str(x.get("attack_rationale", "")),
                semantic_family=str(x.get("semantic_family", "")),
            ))
        return attacks
