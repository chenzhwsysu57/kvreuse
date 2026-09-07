"""Immutable per-step experiment manifests for adaptive attack/defense runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_ref(path: Path) -> dict[str, str]:
    path = path.resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest}


def write_step(run_dir: Path, step: int, action: str, payload: dict[str, Any]) -> Path:
    if step < 1:
        raise ValueError("step must be >= 1")
    safe_action = "".join(char if char.isalnum() or char in "_-" else "_" for char in action)
    steps = run_dir / "steps"
    steps.mkdir(parents=True, exist_ok=True)
    path = steps / f"step_{step:02d}_{safe_action}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable step manifest: {path}")
    record = {"step": step, "action": action,
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path