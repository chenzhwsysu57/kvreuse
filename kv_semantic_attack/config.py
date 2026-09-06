from dataclasses import dataclass


@dataclass
class LLMConfig:
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url: str = ""
    model: str = "qwen3.8-max"
    temperature: float = 0.8
    max_tokens: int = 4096
    timeout: float = 120.0
    max_retries: int = 1
    enable_thinking: bool = False
