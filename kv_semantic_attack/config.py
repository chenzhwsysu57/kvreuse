from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMConfig:
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url: str = "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.8-max"
    temperature: float = 0.8
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 3

@dataclass
class SelfPlayConfig:
    # 每轮 attacker 生成多少个候选攻击
    attacks_per_round: int = 8
    # 每轮最多保留多少个“真实成立”的攻击
    accepted_attacks_per_round: int = 4
    # defender 每轮生成多少个候选 correction
    defenses_per_round: int = 8
    # 最多自博弈轮数
    max_rounds: int = 8

    # correction 的自然语言长度约束（字符级软约束；真正 token 限制可在 prompt 中控制）
    max_correction_chars: int = 180

    # 防止 attacker 反复生成同一种 attack
    require_attack_diversity: bool = True

    # 是否要求 source full-prefill 也能正确
    require_source_clean_success: bool = True
    # 必须要求 target full-prefill 正确，否则不是 KV reuse 特有 failure
    require_target_clean_success: bool = True

    # 选择 defense 时保留当前 correction，避免“强制更新导致退化”
    keep_incumbent: bool = True

    # 停止条件：连续多少轮没有提升
    patience: int = 2
    # score 至少提升多少才算 improvement
    min_improvement: float = 1e-4

    # replay buffer 最多保留多少个 verified attacks
    max_replay_cases: int = 64

    # attacker/defender prompt 中最多展示多少历史 case，避免上下文爆炸
    max_history_for_attacker: int = 12
    max_history_for_defender: int = 16
