#!/usr/bin/env python3
"""Generate labeled A/B instruction pairs without an LLM or GPU."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kv_semantic_attack.synthetic_tasks import (  # noqa: E402
    GENERATOR_VERSION, LAYOUTS, TASK_TYPES, generate_tasks,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-type", type=int, default=100, help="number of A/B pairs per selected type")
    parser.add_argument("--types", nargs="+", choices=tuple(TASK_TYPES), default=list(TASK_TYPES))
    parser.add_argument("--count", action="append", default=[], metavar="TYPE=N", help="override count of one selected type; repeatable")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--min-rows", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=12)
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    parser.add_argument("--direct-option-format-switch", action="store_true",
                        help="clean control: format_switch prefix directly selects JSON/CSV option, never a raw payload")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "kv_semantic_attack/generated_tasks")
    parser.add_argument("--overwrite", action="store_true", help="replace a previous generator output (refuses unrelated files)")
    parser.add_argument("--list-types", action="store_true")
    args = parser.parse_args(argv)
    if args.list_types:
        for name, (dimension, label) in TASK_TYPES.items():
            print(f"{name:24} {dimension:12} {label}")
        return 0
    if args.per_type < 0:
        parser.error("--per-type must be non-negative")
    counts = {name: args.per_type for name in args.types}
    for entry in args.count:
        try:
            name, value = entry.split("=", 1)
            if name not in counts or int(value) < 0:
                raise ValueError
            counts[name] = int(value)
        except ValueError:
            parser.error(f"invalid --count {entry!r}; use a selected TYPE and a non-negative integer")
    if not sum(counts.values()):
        parser.error("at least one pair must be requested")
    if args.direct_option_format_switch and set(counts) != {"format_switch"}:
        parser.error("--direct-option-format-switch requires --types format_switch")

    output = args.output_dir
    if output.exists() and not output.is_dir():
        parser.error(f"output is not a directory: {output}")
    existing = list(output.iterdir()) if output.exists() else []
    generated_names = {f"{name}.jsonl" for name in TASK_TYPES} | {"all.jsonl", "manifest.json"}
    if existing and not args.overwrite:
        parser.error(f"output directory is not empty: {output}; choose a new directory or use --overwrite")
    if existing and any(path.name not in generated_names or not path.is_file() for path in existing):
        parser.error("refusing to overwrite a directory containing unrelated files")

    try:
        records = list(generate_tasks(counts, seed=args.seed, min_rows=args.min_rows,
                                      max_rows=args.max_rows, layouts=tuple(args.layouts),
                                      prompt_protocol=("direct_option_format" if args.direct_option_format_switch
                                                       else "raw_payload_then_option")))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    # Generate and validate everything before replacing previous output files.
    output.mkdir(parents=True, exist_ok=True)
    for path in existing:
        path.unlink()
    grouped = {name: [] for name in counts}
    for record in records:
        grouped[record["attack_type"]].append(record)
    for name, rows in {**grouped, "all": records}.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "generator_version": GENERATOR_VERSION, "seed": args.seed,
        "pairs": len(records), "reuse_directions": 2 * len(records),
        "counts": counts, "min_rows": args.min_rows, "max_rows": args.max_rows,
        "layouts": args.layouts,
        "prompt_protocol": "direct_option_format" if args.direct_option_format_switch else "raw_payload_then_option",
        "observed_layouts": dict(Counter(row["metadata"]["layout"] for row in records)),
        "unique_shared_data": len({row["shared_data_id"] for row in records}),
        "answer_mode": "raw", "split_group_key": "shared_data_id",
        "notes": [
            "These are valid conflict candidates, not model-verified reuse failures.",
            "Keep all variants/directions of shared_data_id in the same split.",
            "No tokenizer was loaded; check A/B shared-block token equality at inference.",
            "Use raw final-output scoring; boxed prefixes and lowercase normalization are incompatible.",
            "The current batch_eval.py requires adaptation before evaluating these tasks.",
            "Never include gold, rules, or metadata in the model input.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(records)} A/B pairs ({len(counts)} types; {2 * len(records)} reuse directions).")
    print(f"Output: {output.resolve()}")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print("NOTE: raw-output evaluation is required; no model inference or token-alignment check was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())