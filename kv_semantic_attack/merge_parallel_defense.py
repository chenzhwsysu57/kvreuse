"""Validate and join parallel Full/Reuse reports without additional inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kv_semantic_attack.adaptive_defense import parse_candidates
from kv_semantic_attack.run_journal import file_ref


def merge_reports(output_dir: Path, source: Path, candidates_path: Path) -> dict:
    from kv_semantic_attack.eval_synthetic_batch import summarize

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    dispatch = read(output_dir / "dispatch_manifest.json")
    candidates = parse_candidates(read(candidates_path))
    expected_jobs = {"full", "direct_reuse", *(c.candidate_id for c in candidates)}
    jobs = dispatch["jobs"]
    if (len(jobs) != len(expected_jobs) or {j["name"] for j in jobs} != expected_jobs
            or any(j["returncode"] != 0 for j in jobs)):
        raise ValueError("missing, duplicate or failed parallel jobs")
    if dispatch["input"] != file_ref(source) or dispatch["candidates"] != file_ref(candidates_path):
        raise ValueError("parallel input/candidate fingerprints differ")
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    expected = {(r["task_id"], side) for r in records for side in ("a", "b")}
    if not records or len(expected) != 2 * len(records):
        raise ValueError("empty or duplicate input tasks")
    full = read(output_dir / "full.json")

    def validate(data, method, suffix):
        if data.get("complete") is not True:
            raise ValueError("incomplete parallel report")
        config = data["config"]
        if config["input_sha256"] != dispatch["input"]["sha256"]:
            raise ValueError("report input fingerprint differs")
        for key in ("model", "batch_size", "max_new_tokens"):
            if config[key] != dispatch[key]:
                raise ValueError(f"report config differs: {key}")
        for key in ("effective_input_sha256", "reasoning", "boxed_output", "output_protocol",
                    "strip_shared_data_tags", "verify_per_type", "torch_version", "model_path"):
            if config[key] != full["config"][key]:
                raise ValueError(f"paired report config differs: {key}")
        if config["methods"] != [method] or config["post_block_suffix"] != suffix.strip():
            raise ValueError("wrong method or suffix in report")
        if method == "reuse" and config["reuse_engine"] != "scatter":
            raise ValueError("parallel reuse must use scatter")
        rows = data["results"]
        indexed = {(r["task_id"], r["target"]): r for r in rows}
        if len(rows) != len(expected) or indexed.keys() != expected:
            raise ValueError("missing or duplicate evaluation directions")
        if any(method not in row for row in rows):
            raise ValueError("missing method results")
        return indexed

    full_rows = validate(full, "full", "")

    def paired(data, suffix):
        reuse_rows = validate(data, "reuse", suffix)
        for key, row in reuse_rows.items():
            if row["attack_type"] != full_rows[key]["attack_type"] or row["source_gold"] != full_rows[key]["source_gold"]:
                raise ValueError("paired row metadata differs")
        rows = [{**row, "reuse": reuse_rows[key]["reuse"]} for key, row in full_rows.items()]
        return {"results": rows, "summary": summarize(rows), "timing_seconds": data["timing_seconds"]}

    return {"complete": True, "model": dispatch["model"], "input": str(source),
            "candidate_file": str(candidates_path), "reuse_engine": "scatter", "dispatch": dispatch,
            "full": full, "direct_reuse": paired(read(output_dir / "direct_reuse.json"), ""),
            "candidates": [{"candidate": c.__dict__,
                            **paired(read(output_dir / f"suffix_{c.candidate_id}.json"), c.suffix)} for c in candidates]}


def write_merged(output_dir: Path, source: Path, candidates: Path, output: Path) -> None:
    report = merge_reports(output_dir, source, candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_merged(args.output_dir, args.input, args.candidates, args.output)


if __name__ == "__main__":
    main()