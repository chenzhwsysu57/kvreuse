#!/usr/bin/env python3
"""Pre-download and validate the Qwen3 checkpoints used by kvreuse.

Run this on a network-enabled login/data node so GPU jobs do not spend their
allocation waiting for ModelScope downloads.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from modelscope import snapshot_download


MODEL_IDS = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
    "8b": "Qwen/Qwen3-8B",
}


def validate_model_files(model_dir: Path) -> None:
    """Fail early if a sharded checkpoint has missing or partial files."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        missing = sorted(
            {name for name in index["weight_map"].values() if not (model_dir / name).is_file()}
        )
        if missing:
            raise RuntimeError(f"missing checkpoint shards in {model_dir}: {missing}")
    partial = sorted(model_dir.glob("*.incomplete"))
    if partial:
        raise RuntimeError(f"incomplete checkpoint files in {model_dir}: {partial}")
    if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("pytorch_model*.bin")):
        raise RuntimeError(f"no model weight files found in {model_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_IDS), default=sorted(MODEL_IDS),
        help="model sizes to download (default: all four)",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(os.environ["MODELSCOPE_CACHE"]) if os.environ.get("MODELSCOPE_CACHE") else None,
        help="ModelScope cache root; defaults to MODELSCOPE_CACHE or ModelScope's default",
    )
    parser.add_argument("--revision", default=None, help="optional ModelScope revision")
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    for size in args.models:
        model_id = MODEL_IDS[size]
        print(f"[{size}] downloading {model_id}", flush=True)
        path = Path(
            snapshot_download(
                model_id,
                revision=args.revision,
                cache_dir=args.cache_dir,
                max_workers=args.max_workers,
            )
        )
        validate_model_files(path)
        print(f"[{size}] ready: {path}", flush=True)
    print("all requested models are ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
