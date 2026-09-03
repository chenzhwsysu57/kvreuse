import os

from kv_semantic_selfplay import (
    Attacker,
    ChatLLM,
    Defender,
    LLMConfig,
    SelfPlayConfig,
    SelfPlayOrchestrator,
)
from kv_semantic_selfplay.mock_executor import MockExecutor


def main():
    llm_cfg = LLMConfig(
        # 替换 {WorkspaceId}
        base_url="https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        model="qwen3.8-max",
    )
    sp_cfg = SelfPlayConfig(
        attacks_per_round=8,
        accepted_attacks_per_round=4,
        defenses_per_round=8,
        max_rounds=6,
    )

    llm = ChatLLM(llm_cfg)
    attacker = Attacker(llm)
    defender = Defender(llm)

    # TODO: 换成你仓库里的真实 Qwen3-1.7B + KV reuse + RoPE repair executor
    executor = MockExecutor()

    runner = SelfPlayOrchestrator(
        attacker=attacker,
        defender=defender,
        executor=executor,
        config=sp_cfg,
        log_dir="./selfplay_logs",
    )

    final_correction = runner.run(initial_correction="")
    print("Final correction:")
    print(final_correction)


if __name__ == "__main__":
    main()
