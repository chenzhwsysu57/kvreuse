"""
仅用于测试 orchestration 能否跑通。
真正实验时删掉/替换成你仓库里的 Qwen3 executor。
"""

from .executor import KVReuseExecutor
from .schemas import ExecResult


class MockExecutor(KVReuseExecutor):
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
        # 这里只是假结果，不能用于实验。
        return ExecResult(
            text=expected_answer,
            score=1.0,
            passed=True,
            metadata={"mock": True},
        )

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
        # 示例：空 correction 时人为失败；非空时人为成功。
        if correction.strip():
            return ExecResult(
                text=expected_answer,
                score=1.0,
                passed=True,
                metadata={"mock": True},
            )
        return ExecResult(
            text="<mock failure>",
            score=0.0,
            passed=False,
            metadata={"mock": True},
        )
