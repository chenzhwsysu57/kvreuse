from .executor import Executor
from .schemas import ExecutionResult
from .step_logger import StepLogger


class MockExecutor(Executor):
    def __init__(self, step_logger: StepLogger | None = None):
        self.step_logger = step_logger

    def _log(self, kind: str, inputs: dict[str, str], result: ExecutionResult) -> None:
        if self.step_logger is None:
            return
        input_text = "\n\n".join(
            f"===== {name} =====\n{text}" for name, text in inputs.items()
        )
        self.step_logger.write(
            f"executor_{kind}",
            f"mode: mock\n\n{input_text}\n\n===== MODEL OUTPUT =====\n{result.text}\n",
        )

    def run_full(self, *, prefix, shared_block, question, gold, label):
        result = ExecutionResult(
            text=f"Reasoning follows {label}. \\boxed{{A}}",
            metadata={"label": label, "mock": True},
        )
        self._log(label, {
            "prefix": prefix, "shared_block": shared_block,
            "question": question, "gold": gold,
        }, result)
        return result

    def run_reuse(self, *, donor_prefix, target_prefix, shared_block,
                  question, remedy, gold, label):
        text = f"Reasoning follows target task. remedy={remedy} \\boxed{{A}}"
        result = ExecutionResult(text=text, metadata={"label": label, "mock": True})
        self._log(label, {
            "donor_prefix": donor_prefix, "target_prefix": target_prefix,
            "shared_block": shared_block, "question": question,
            "remedy": remedy, "gold": gold,
        }, result)
        return result
