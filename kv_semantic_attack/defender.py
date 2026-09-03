import json
from typing import List, Sequence

from .llm_client import ChatLLM
from .schemas import DefenseCandidate, VerifiedAttack


DEFENDER_SYSTEM_PROMPT = """
You are a semantic correction synthesizer for a KV-reuse system.

You observe execution traces from a target language model. Your job is to
propose short natural-language instructions that can be inserted AFTER the
reused KV block to improve the target model's behavior.

Important:
1. Start from the observed failures; do not assume a predefined mechanism.
2. Do not modify the KV cache, source task, target task, shared block, or query.
3. Do not ask for recomputation.
4. The correction must be task-independent and reusable across unseen tasks.
5. It must preserve useful information in the reused shared block.
6. Avoid task-specific answers, examples, or dataset-specific wording.
7. Prefer short, semantically precise corrections.
8. Return multiple diverse hypotheses; the execution system will decide which is best.
9. Return JSON only.
""".strip()


class Defender:
    def __init__(self, llm: ChatLLM):
        self.llm = llm

    def propose(
        self,
        *,
        incumbent: str,
        failures: Sequence[VerifiedAttack],
        n: int,
        max_chars: int,
    ) -> List[DefenseCandidate]:
        cases = []
        for item in failures:
            if not item.accepted:
                continue
            a = item.attack
            cases.append({
                "source_instruction": a.source_instruction,
                "target_instruction": a.target_instruction,
                "shared_block": a.shared_block,
                "query": a.query,
                "expected_target_answer": a.expected_target_answer,
                "full_prefill_target_output": item.target_clean.text,
                "kv_reuse_output": item.reuse_result.text,
                "failure_strength": item.failure_strength,
                "semantic_family": a.semantic_family,
            })

        user_prompt = f"""
Current correction:
{incumbent if incumbent else "<EMPTY>"}

Verified KV-reuse failures:
{json.dumps(cases, ensure_ascii=False, indent=2)}

First infer the common failure mechanisms from the traces.
Then propose {n} revised post-reuse corrections.

Do NOT merely paraphrase the incumbent.
The candidates should explore meaningfully different semantic hypotheses.

Each candidate must be at most {max_chars} characters.

Return:
{{
  "diagnosis": "short shared diagnosis",
  "candidates": [
    {{
      "text": "...",
      "rationale": "why this may fix the observed failures"
    }}
  ]
}}
""".strip()

        data = self.llm.chat_json([
            {"role": "system", "content": DEFENDER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])

        diagnosis = str(data.get("diagnosis", "")) if isinstance(data, dict) else ""
        raw = data.get("candidates", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            raise ValueError("Defender candidates must be a JSON list.")

        out: List[DefenseCandidate] = []
        for x in raw[:n]:
            text = str(x["text"]).strip()
            if len(text) > max_chars:
                continue
            out.append(DefenseCandidate(
                text=text,
                diagnosis=diagnosis,
                rationale=str(x.get("rationale", "")),
                parent=incumbent,
            ))
        return out
