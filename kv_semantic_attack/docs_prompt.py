"""Build role prompts and inject the latest self-play state."""

from pathlib import Path


DOCS = Path(__file__).parent / "docs"


def _load(name: str) -> str:
    text = (DOCS / name).read_text(encoding="utf-8").strip()
    marker = "### BACKGROUND ###"
    return text[text.index(marker):] if marker in text else text


def _history_block(attack_history: list[str], defense_history: list[str]) -> str:
    """Render the mutable history field shared by attacker and defender."""
    attack = "\n\n".join(attack_history) or "暂时为空"
    defense = "\n\n".join(defense_history) or "暂时为空"
    return (
        "### 历史摘要 ###\n"
        "Attack Judger:\n"
        f"{attack}\n\n"
        "Defend Judger:\n"
        f"{defense}"
    )


def _inject_state(text: str, attack_history: list[str],
                  defense_history: list[str]) -> str:
    marker = "### 返回要求 ###"
    state = _history_block(attack_history, defense_history)
    return text.replace(marker, f"{state}\n\n{marker}", 1)


def attacker_prompt(attack_history: list[str], defense_history: list[str]) -> str:
    return _inject_state(
        _load("attacker.txt"), attack_history, defense_history
    )


def defender_prompt(attack_history: list[str], defense_history: list[str]) -> str:
    return _inject_state(
        _load("defender.txt"), attack_history, defense_history
    )


def judger_prompt(name: str, execution_info: str) -> str:
    text = _load(name)
    start = "### EXECUTOR INFO ###"
    end = "### 返回格式 ###"
    if start not in text or end not in text:
        raise ValueError(f"{name} has no Executor section")
    return text[:text.index(start)] + start + "\n" + execution_info.strip() + "\n\n" + text[text.index(end):]
