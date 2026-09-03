from abc import ABC, abstractmethod
from typing import Iterable, List

from .schemas import AttackCase, DefenseCandidate, ExecResult, VerifiedAttack


class KVReuseExecutor(ABC):
    """
    你现有 Qwen3 / ModelScope / KV reuse / RoPE repair 代码只需要实现这个接口。

    重要：
    - attacker / defender 不应该知道 KV reuse 的实现细节。
    - 所有真实性判断都通过这里完成。
    """

    @abstractmethod
    def run_full_prefill(
        self,
        *,
        instruction: str,
        shared_block: str,
        query: str,
        expected_answer: str,
        eval_type: str,
        eval_metadata: dict,
    ) -> ExecResult:
        """
        正常 full-prefill，不做 KV reuse。
        用于验证 task 本身是否是 target model 能做对的。
        """
        raise NotImplementedError

    @abstractmethod
    def run_kv_reuse(
        self,
        *,
        source_instruction: str,
        target_instruction: str,
        shared_block: str,
        query: str,
        correction: str,
        expected_answer: str,
        eval_type: str,
        eval_metadata: dict,
    ) -> ExecResult:
        """
        真正执行：
            source_instruction -> shared_block 的 KV
            复用到 target_instruction / query
            应用你的 RoPE 修复
            在 shared KV 之后插入 correction
        """
        raise NotImplementedError

    def verify_attack(
        self,
        attack: AttackCase,
        correction: str,
        *,
        require_source_clean_success: bool = True,
        require_target_clean_success: bool = True,
    ) -> VerifiedAttack:
        source_clean = None
        if require_source_clean_success:
            source_clean = self.run_full_prefill(
                instruction=attack.source_instruction,
                shared_block=attack.shared_block,
                query=attack.query,
                expected_answer=attack.expected_source_answer,
                eval_type=attack.eval_type,
                eval_metadata=attack.eval_metadata,
            )

        target_clean = self.run_full_prefill(
            instruction=attack.target_instruction,
            shared_block=attack.shared_block,
            query=attack.query,
            expected_answer=attack.expected_target_answer,
            eval_type=attack.eval_type,
            eval_metadata=attack.eval_metadata,
        )

        reuse_result = self.run_kv_reuse(
            source_instruction=attack.source_instruction,
            target_instruction=attack.target_instruction,
            shared_block=attack.shared_block,
            query=attack.query,
            correction=correction,
            expected_answer=attack.expected_target_answer,
            eval_type=attack.eval_type,
            eval_metadata=attack.eval_metadata,
        )

        if require_source_clean_success and source_clean is not None and not source_clean.passed:
            return VerifiedAttack(
                attack=attack,
                source_clean=source_clean,
                target_clean=target_clean,
                reuse_result=reuse_result,
                accepted=False,
                reject_reason="source_clean_failed",
            )

        if require_target_clean_success and not target_clean.passed:
            return VerifiedAttack(
                attack=attack,
                source_clean=source_clean,
                target_clean=target_clean,
                reuse_result=reuse_result,
                accepted=False,
                reject_reason="target_clean_failed",
            )

        # 核心 gate：clean target 成功，而 reuse 失败
        accepted = target_clean.passed and (not reuse_result.passed)
        return VerifiedAttack(
            attack=attack,
            source_clean=source_clean,
            target_clean=target_clean,
            reuse_result=reuse_result,
            accepted=accepted,
            reject_reason="" if accepted else "reuse_did_not_fail",
        )

    def score_correction(
        self,
        correction: str,
        attacks: Iterable[AttackCase],
    ) -> tuple[float, float]:
        """
        默认返回 (mean_score, worst_case_score)。
        可在你自己的 executor 中 override，加入 task-family macro average 等。
        """
        scores: List[float] = []
        for attack in attacks:
            result = self.run_kv_reuse(
                source_instruction=attack.source_instruction,
                target_instruction=attack.target_instruction,
                shared_block=attack.shared_block,
                query=attack.query,
                correction=correction,
                expected_answer=attack.expected_target_answer,
                eval_type=attack.eval_type,
                eval_metadata=attack.eval_metadata,
            )
            scores.append(result.score)

        if not scores:
            return 0.0, 0.0
        return sum(scores) / len(scores), min(scores)
