"""Fast, non-model connectivity check for the configured compatible API."""

import os
from pathlib import Path
from urllib.parse import urlparse


def load_local_env(path: Path = Path(".env.local")) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    print("[1/5] reading .env.local", flush=True)
    load_local_env()
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    model = os.getenv("DASHSCOPE_MODEL", "")
    if not api_key:
        print("FAIL: DASHSCOPE_API_KEY is empty", flush=True)
        return 2
    if not base_url:
        print("FAIL: DASHSCOPE_BASE_URL is empty", flush=True)
        return 2
    host = urlparse(base_url).hostname or "<invalid-url>"
    print(f"[2/5] config found: key=present, host={host}, model={model or '<empty>'}", flush=True)

    print("[3/5] importing OpenAI client", flush=True)
    from openai import OpenAI
    print("[4/5] constructing client", flush=True)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=15.0, max_retries=0)
    print("[5/5] sending minimal JSON request", flush=True)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个测试助手，请直接回答用户问题。"},
            {"role": "user", "content": "请用一句中文说明：KV cache reuse 可能为什么会造成语义错误？"},
        ],
        temperature=0,
        max_tokens=16,
        extra_body={"enable_thinking": False},
    )
    content = response.choices[0].message.content
    if not content:
        print("FAIL: API returned empty content", flush=True)
        return 1
    print(f"SUCCESS: API responded; content_length={len(content)}", flush=True)
    print("----- model response text -----", flush=True)
    print(content, flush=True)
    print("----- end model response -----", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
