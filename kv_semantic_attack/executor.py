from abc import ABC, abstractmethod
from .schemas import AttackCase, ExecutionResult


class Executor(ABC):
    @abstractmethod
    def run_full(self, *, prefix: str, shared_block: str, question: str,
                 gold: str, label: str) -> ExecutionResult:
        raise NotImplementedError

    @abstractmethod
    def run_reuse(self, *, donor_prefix: str, target_prefix: str,
                  shared_block: str, question: str, remedy: str,
                  gold: str, label: str) -> ExecutionResult:
        raise NotImplementedError

    def execute_attack(self, case: AttackCase, remedy: str) -> dict[str, ExecutionResult]:
        return {
            "full_a": self.run_full(prefix=case.prefix_a, shared_block=case.shared_block,
                                     question=case.question, gold=case.gold_a, label="full_a"),
            "full_b": self.run_full(prefix=case.prefix_b, shared_block=case.shared_block,
                                     question=case.question, gold=case.gold_b, label="full_b"),
            "reuse_b_to_a": self.run_reuse(donor_prefix=case.prefix_b, target_prefix=case.prefix_a,
                                             shared_block=case.shared_block, question=case.question,
                                             remedy=remedy, gold=case.gold_a, label="reuse_b_to_a"),
            "reuse_a_to_b": self.run_reuse(donor_prefix=case.prefix_a, target_prefix=case.prefix_b,
                                             shared_block=case.shared_block, question=case.question,
                                             remedy=remedy, gold=case.gold_b, label="reuse_a_to_b"),
        }

    def execute_attack_batch(self, cases: list[AttackCase], remedy: str) -> list[dict[str, ExecutionResult]]:
        """Batch hook.  Backends may override this; the safe fallback is serial."""
        return [self.execute_attack(case, remedy) for case in cases]

    def execute_defense(self, case: AttackCase, remedy: str,
                        without_remedy: dict[str, ExecutionResult] | None = None):
        if without_remedy is None:
            without_remedy = {
                "reuse_b_to_a": self.run_reuse(
                    donor_prefix=case.prefix_b, target_prefix=case.prefix_a,
                    shared_block=case.shared_block, question=case.question,
                    remedy="", gold=case.gold_a, label="reuse_b_to_a_without_remedy"),
                "reuse_a_to_b": self.run_reuse(
                    donor_prefix=case.prefix_a, target_prefix=case.prefix_b,
                    shared_block=case.shared_block, question=case.question,
                    remedy="", gold=case.gold_b, label="reuse_a_to_b_without_remedy"),
            }
        return {
            "without_remedy": without_remedy,
            "with_remedy": {
                "reuse_b_to_a": self.run_reuse(donor_prefix=case.prefix_b, target_prefix=case.prefix_a,
                                                 shared_block=case.shared_block, question=case.question,
                                                 remedy=remedy, gold=case.gold_a, label="reuse_b_to_a_with_remedy"),
                "reuse_a_to_b": self.run_reuse(donor_prefix=case.prefix_a, target_prefix=case.prefix_b,
                                                 shared_block=case.shared_block, question=case.question,
                                                 remedy=remedy, gold=case.gold_b, label="reuse_a_to_b_with_remedy"),
            },
        }
