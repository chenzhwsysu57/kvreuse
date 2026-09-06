import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - allows offline mock tests
    OpenAI = None

from .config import LLMConfig
from .step_logger import StepLogger


class ChatLLM:
    """
    对 OpenAI-compatible Chat Completions API 的薄封装。
    适配百炼 / DashScope compatible-mode。

    设计目标：
    1. attacker / defender 只依赖这一层，不耦合具体供应商。
    2. 尽量要求 JSON 输出；若 provider 不支持 response_format，则仍可从文本中提取 JSON。
    3. API key 不写入代码。
    """

    def __init__(self, config: LLMConfig, trace_dir: Optional[str] = None,
                 debug: bool = False, step_logger: StepLogger | None = None):
        self.config = config
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.step_logger = step_logger
        self.debug = debug
        self._trace_index = 0
        if OpenAI is None:
            raise RuntimeError(
                "ChatLLM requires the 'openai' package; offline tests can use "
                "a fake chat_json provider instead."
            )
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
        trace_name: str = "llm",
    ) -> str:
        kwargs: Dict[str, Any] = dict(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            extra_body={"enable_thinking": self.config.enable_thinking},
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        if self.debug:
            print("\n===== LLM REQUEST (%s) =====" % trace_name, flush=True)
            for message in messages:
                print(f"[{message['role']}]\n{message['content']}\n", flush=True)

        try:
            completion = self.client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
        except Exception as exc:
            self._save_trace(trace_name, messages, f"[API ERROR] {exc!r}")
            raise
        if content is None:
            raise RuntimeError("LLM returned empty content.")
        if self.debug:
            print("===== LLM RAW RESPONSE =====", flush=True)
            print(content, flush=True)
            print("===== END LLM CALL =====\n", flush=True)
        self._save_trace(trace_name, messages, content)
        return content

    def _save_trace(self, name: str, messages: List[Dict[str, str]], response: str) -> None:
        if self.trace_dir is None and self.step_logger is None:
            return
        stamp = datetime.now().astimezone().isoformat()
        prompt_text = "\n\n".join(
            f"[{message['role']}]\n{message['content']}" for message in messages
        )
        content = (
            f"timestamp: {stamp}\nmodel: {self.config.model}\n"
            f"base_url: {self.config.base_url}\n"
            f"temperature: {self.config.temperature}\n"
            f"max_tokens: {self.config.max_tokens}\n"
            f"enable_thinking: {self.config.enable_thinking}\n\n"
            f"===== INPUT MESSAGES =====\n{prompt_text}\n\n"
            f"===== RAW MODEL OUTPUT =====\n{response}\n"
        )
        if self.step_logger is not None:
            self.step_logger.write(name, content)
        else:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_index += 1
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            path = self.trace_dir / f"{self._trace_index:04d}_{safe_name}.txt"
            path.write_text(content, encoding="utf-8")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        try_native_json: bool = False,
        trace_name: str = "llm",
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
                    trace_name=trace_name,
                )
                return json.loads(text)
            except Exception:
                pass

        text = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            trace_name=trace_name,
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
