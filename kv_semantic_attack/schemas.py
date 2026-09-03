from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AttackCase:
    """
    attacker 生成的 synthetic counterexample proposal。

    shared_block 必须在 source/target 两个条件中保持完全相同。
    source_instruction 与 target_instruction 表示冲突的 task semantics。
    """
    attack_id: str
    source_instruction: str
    target_instruction: str
    shared_block: str
    query: str

    # attacker 给出的“预期答案”仅用于构造与 sanity check。
    # 是否真正正确，仍由 clean execution / evaluator 验证。
    expected_source_answer: str
    expected_target_answer: str

    # exact / contains / regex / llm_judge / custom
    eval_type: str = "exact"
    eval_metadata: Dict[str, Any] = field(default_factory=dict)

    # 便于分析，不参与真实评分
    attack_rationale: str = ""
    semantic_family: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecResult:
    text: str
    score: float
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifiedAttack:
    attack: AttackCase

    # source task clean full-prefill
    source_clean: Optional[ExecResult]

    # target task clean full-prefill
    target_clean: ExecResult

    # 当前 correction 下的 KV reuse
    reuse_result: ExecResult

    # 是否被系统 gate 接受为“真实 KV-reuse counterexample”
    accepted: bool
    reject_reason: str = ""

    @property
    def failure_strength(self) -> float:
        # clean target 越好、reuse 越差，counterexample 越强
        return max(0.0, self.target_clean.score - self.reuse_result.score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attack": self.attack.to_dict(),
            "source_clean": None if self.source_clean is None else asdict(self.source_clean),
            "target_clean": asdict(self.target_clean),
            "reuse_result": asdict(self.reuse_result),
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
            "failure_strength": self.failure_strength,
        }


@dataclass
class DefenseCandidate:
    text: str
    diagnosis: str = ""
    rationale: str = ""
    parent: str = ""
    score: Optional[float] = None
    worst_case_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoundRecord:
    round_idx: int
    incumbent_before: str
    attack_proposals: List[AttackCase]
    verified_attacks: List[VerifiedAttack]
    defense_candidates: List[DefenseCandidate]
    incumbent_after: str
    incumbent_score: float
