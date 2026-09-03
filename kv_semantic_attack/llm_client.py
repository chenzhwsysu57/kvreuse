import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import LLMConfig


class ChatLLM:
    """
    对 OpenAI-compatible Chat Completions API 的薄封装。
    适配百炼 / DashScope compatible-mode。

    设计目标：
    1. attacker / defender 只依赖这一层，不耦合具体供应商。
    2. 尽量要求 JSON 输出；若 provider 不支持 response_format，则仍可从文本中提取 JSON。
    3. API key 不写入代码。
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {config.api_key_env!r} is not set."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        kwargs: Dict[str, Any] = dict(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        completion = self.client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty content.")
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        try_native_json: bool = False,
    ) -> Any:
        """
        默认不强依赖 provider 的 structured-output 支持。
        若 try_native_json=True，会尝试 response_format={"type":"json_object"}。
        """
        if try_native_json:
            try:
                text = self.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                return json.loads(text)
            except Exception:
                pass

        text = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> Any:
        text = text.strip()

        # 1) 直接 JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2) ```json ... ```
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 3) 尝试抓最外层 [] 或 {}
        candidates = []
        if "[" in text and "]" in text:
            candidates.append(text[text.find("["): text.rfind("]") + 1])
        if "{" in text and "}" in text:
            candidates.append(text[text.find("{"): text.rfind("}") + 1])

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Could not parse JSON from LLM output:\n{text}")
