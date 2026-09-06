"""Data records for the documentation-defined semantic attack game."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AttackCase:
    prefix_a: str
    prefix_b: str
    shared_block: str
    question: str
    gold_a: str
    gold_b: str
    case_id: str = field(default="")
    # Private planning trace; executor and judgers use only the task fields.
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionResult:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttackJudgment:
    summary: str
    attack_success: bool
    raw_output: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DefenseProposal:
    reasoning: str
    remedy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DefenseJudgment:
    summary: str
    defense_success: bool
    raw_output: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttackRecord:
    case: AttackCase
    executions: dict[str, ExecutionResult]
    judgment: AttackJudgment
    attempt: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "executions": {k: v.to_dict() for k, v in self.executions.items()},
            "judgment": self.judgment.to_dict(),
            "attempt": self.attempt,
        }


@dataclass
class DefenseRecord:
    proposal: DefenseProposal
    evaluations: dict[str, ExecutionResult]
    judgment: DefenseJudgment
    attempt: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "evaluations": {k: v.to_dict() for k, v in self.evaluations.items()},
            "judgment": self.judgment.to_dict(),
            "attempt": self.attempt,
        }


@dataclass
class RoundRecord:
    round_idx: int
    remedy_before: str
    attack_submissions: list[dict[str, Any]]
    accepted_attack: dict[str, Any] | None
    defense_submissions: list[dict[str, Any]]
    remedy_after: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
