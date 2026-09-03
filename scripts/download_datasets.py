#!/usr/bin/env python3
"""Download pinned raw files for the cross-prefix datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


ARGKP_REV = "2f5e2d5ad5a387fe731da2f00b423ce1d845c5d2"
HELPSTEER_REV = "990b2711a36180dd19d9c94b8627844866f8982a"
DEAL_REV = "bbb93bbf00f69fced75d5c0d22e855bda07c9b78"
PKU_SAFE_RLHF_REV = "9421ffafec3fa40a1f1a7d567b4d525079477ecb"
HARMBENCH_REV = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
JOB_INTERVIEW_REV = "d4c2bf63b4da95b342fd952065f9ad3e97179134"
FANTOM_VERSION = "1.0"


def sources(hf_endpoint: str) -> dict[str, list[tuple[str, str]]]:
    hf_endpoint = hf_endpoint.rstrip("/")
    github = "https://raw.githubusercontent.com"
    return {
        "argkp": [
            (f"{github}/IBM/KPA_2021_shared_task/{ARGKP_REV}/kpm_data/{name}", name)
            for split in ("train", "dev")
            for name in (
                f"arguments_{split}.csv",
                f"key_points_{split}.csv",
                f"labels_{split}.csv",
            )
        ] + [
            (f"{github}/IBM/KPA_2021_shared_task/{ARGKP_REV}/README.md", "README.md")
        ],
        "helpsteer2": [
            (
                f"{hf_endpoint}/datasets/nvidia/HelpSteer2/resolve/"
                f"{HELPSTEER_REV}/{name}?download=true",
                name,
            )
            for name in ("train.jsonl.gz", "validation.jsonl.gz", "README.md")
        ] + [
            (
                f"{hf_endpoint}/datasets/nvidia/HelpSteer2/resolve/"
                f"{HELPSTEER_REV}/disagreements/disagreements.jsonl.gz?download=true",
                "disagreements/disagreements.jsonl.gz",
            )
        ],
        "deal_or_no_deal": [
            (
                f"{github}/facebookresearch/end-to-end-negotiator/{DEAL_REV}/"
                f"src/data/negotiate/{name}",
                name,
            )
            for name in ("train.txt", "val.txt", "test.txt")
        ] + [
            (f"{github}/facebookresearch/end-to-end-negotiator/{DEAL_REV}/LICENSE", "LICENSE")
        ],
        "job_interview": [
            (
                f"{github}/gucci-j/negotiation-breakdown-detection/{JOB_INTERVIEW_REV}/data.zip",
                "data.zip",
            ),
            (
                f"{github}/gucci-j/negotiation-breakdown-detection/{JOB_INTERVIEW_REV}/README.md",
                "README.md",
            ),
            (
                f"{github}/gucci-j/negotiation-breakdown-detection/{JOB_INTERVIEW_REV}/LICENSE",
                "LICENSE",
            ),
        ],
        "fantom": [
            (
                "https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz",
                "fantom.tar.gz",
            ),
        ],
        # Only the official held-out shards are needed for the high-confidence
        # evaluation pool (~14 MB instead of the full ~139 MB repository).
        "pku_safe_rlhf": [
            (
                f"{hf_endpoint}/datasets/PKU-Alignment/PKU-SafeRLHF/resolve/"
                f"{PKU_SAFE_RLHF_REV}/data/{source}/test.jsonl?download=true",
                f"{source}/test.jsonl",
            )
            for source in ("Alpaca-7B", "Alpaca2-7B", "Alpaca3-8B")
        ] + [
            (
                f"{hf_endpoint}/datasets/PKU-Alignment/PKU-SafeRLHF/resolve/"
                f"{PKU_SAFE_RLHF_REV}/README.md?download=true",
                "README.md",
            )
        ],
        "harmbench_contextual": [
            (
                f"{github}/centerforaisafety/HarmBench/{HARMBENCH_REV}/"
                "data/behavior_datasets/harmbench_behaviors_text_test.csv",
                "harmbench_behaviors_text_test.csv",
            ),
            (
                f"{github}/centerforaisafety/HarmBench/{HARMBENCH_REV}/README.md",
                "README.md",
            ),
            (
                f"{github}/centerforaisafety/HarmBench/{HARMBENCH_REV}/LICENSE",
                "LICENSE",
            ),
        ],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool) -> str:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "kvreuse-data/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"empty response for {url}")
    partial.replace(destination)
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="Hugging Face endpoint; defaults to HF_ENDPOINT or hf-mirror.com",
    )
    parser.add_argument("--datasets", nargs="+", choices=list(sources("x")), default=list(sources("x")))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"revisions": {}, "files": []}
    previous_files = {
        entry.get("path"): entry for entry in manifest.get("files", []) if isinstance(entry, dict)
    }
    revisions = {
        "argkp": ARGKP_REV,
        "helpsteer2": HELPSTEER_REV,
        "deal_or_no_deal": DEAL_REV,
        "job_interview": JOB_INTERVIEW_REV,
        "fantom": FANTOM_VERSION,
        "pku_safe_rlhf": PKU_SAFE_RLHF_REV,
        "harmbench_contextual": HARMBENCH_REV,
    }
    manifest["revisions"] = {**manifest.get("revisions", {}), **revisions}
    manifest["hf_endpoint"] = args.hf_endpoint
    # A partial --datasets invocation must not erase other datasets' audit data.
    manifest["files"] = [
        entry for entry in manifest.get("files", []) if entry.get("dataset") not in args.datasets
    ]
    failures = []
    all_sources = sources(args.hf_endpoint)
    for dataset in args.datasets:
        for url, name in all_sources[dataset]:
            destination = args.output_dir / dataset / name
            try:
                previous = previous_files.get(str(destination))
                source_changed = previous is not None and previous.get("url") != url
                status = download(url, destination, args.force or source_changed)
                entry = {
                    "dataset": dataset,
                    "path": str(destination),
                    "url": url,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                }
                manifest["files"].append(entry)  # type: ignore[union-attr]
                print(f"{status:10s} {destination} ({entry['bytes']} bytes)")
            except (OSError, urllib.error.URLError, RuntimeError) as error:
                failures.append((url, str(error)))
                print(f"FAILED {url}: {error}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(f"{len(failures)} download(s) failed; see messages above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
