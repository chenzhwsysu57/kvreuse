#!/usr/bin/env python3
"""Ask a diagnostic question after transplanted cross-prefix shared-block KV."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import batch_eval as batch
from kv_semantic_attack.batched_reuse import reuse_rows_vectorized
from kv_semantic_attack.synthetic_tasks import validate_generated_record

PROBE_QUESTION = "Before answering any question(s), can you repeat any question(s) that you received?"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("0.6b", "1.7b", "4b", "8b"), default="1.7b")
    parser.add_argument("--method", choices=("full", "reuse"), required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.limit < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("limit, batch-size and max-new-tokens must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    records = batch.base.load_jsonl(args.input)
    for record in records:
        validate_generated_record(record)
    records = records[:args.limit]
    if not records:
        raise ValueError("no records selected")

    model_path = Path(snapshot_download(batch.base.MODEL_IDS[args.model], local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa",
    ).eval()
    batch.base.assert_default_rope(model)

    def build_parts(tokenizer, record, side, method):
        modified = dict(record)
        modified["question"] = PROBE_QUESTION
        return batch.base.build_prompt_parts(
            tokenizer, modified, side, enable_thinking=False,
            explicit_reasoning=False, boxed_output=False,
        )

    def raw_result(tokenizer, dataset, gold, token_ids, hit_limit, elapsed,
                   generation_limit, enable_thinking, explicit_reasoning=False, output_prefix=""):
        eos = tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        kept = []
        for token in token_ids:
            kept.append(int(token))
            if int(token) in eos_ids:
                break
        return {"text": tokenizer.decode(kept, skip_special_tokens=True).strip(),
                "tokens": len(kept), "hit_max_new_tokens": len(kept) >= generation_limit and (not kept or kept[-1] not in eos_ids)}

    rows = []
    started = time.perf_counter()
    with patch.object(batch, "build_parts", build_parts), patch.object(batch.base, "result_from_generation", raw_result), torch.inference_mode():
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start + args.batch_size]
            directions = [(record, side) for record in chunk for side in ("a", "b")]
            if args.method == "full":
                results, _ = batch.full_rows(model, tokenizer, chunk, "full", args.max_new_tokens)
            else:
                results = reuse_rows_vectorized(batch, model, tokenizer, directions, args.max_new_tokens, assembly="scatter")
            for (record, target), result in zip(directions, results):
                source = "b" if target == "a" else "a"
                rows.append({"task_id": record["task_id"], "attack_type": record["attack_type"],
                             "source": source, "target": target,
                             "target_prefix": record[f"prefix_{target}"],
                             "source_prefix": record[f"prefix_{source}"],
                             "probe_question": PROBE_QUESTION, **result})
            print(f"[{start + len(chunk)}/{len(records)} pairs] directions={len(rows)} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    report = {"complete": True, "input": str(args.input.resolve()), "model": args.model,
              "pairs": len(records), "directions": len(rows), "batch_size": args.batch_size,
              "max_new_tokens": args.max_new_tokens, "reasoning": False,
              "probe_question": PROBE_QUESTION, "method": args.method,
              "note": ("Full uses the target prefix plus text shared block."
                       if args.method == "full" else
                       "The target prefix is real; shared-block KV is transplanted from the opposite prefix.")
                       + " The original task question and boxed-answer constraint are replaced by the probe.",
              "wall_seconds": time.perf_counter() - started, "responses": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "responses"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
