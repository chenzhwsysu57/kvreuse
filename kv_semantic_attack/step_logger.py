"""Sequential trace writer shared by all self-play components."""

from pathlib import Path
import re
import threading


class StepLogger:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = [
            int(path.name.split("_", 1)[0])
            for path in self.directory.glob("[0-9][0-9][0-9][0-9]_*.txt")
            if path.name.split("_", 1)[0].isdigit()
        ]
        self._step = max(existing, default=0)
        self._lock = threading.Lock()

    def write(self, name: str, content: str) -> Path:
        with self._lock:
            self._step += 1
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "step"
            path = self.directory / f"{self._step:04d}_{safe_name}.txt"
            path.write_text(content, encoding="utf-8")
            return path
