#!/usr/bin/env python3
"""Evaluate Full once and multiple universal suffixes on one attack batch."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import batch_eval as batch  # noqa: E402
from kv_semantic_attack.batched_reuse import reuse_rows_vectorized  # noqa: E402
from kv_semantic_attack.adaptive_defense import parse_candidates  # noqa: E402
from kv_semantic_attack.eval_synthetic_batch import raw_batch_adapter, raw_result, summarize  # noqa: E402
from kv_semantic_attack.synthetic_tasks import score_response, validate_generated_record  # noqa: E402
from kv_semantic_attack.run_journal import file_ref, write_step  # noqa: E402


def _progress(phase: str, completed: int, total: int, started: float) -> None:
    memory = ""
    if batch.torch.cuda.is_available():
        allocated = batch.torch.cuda.memory_allocated() / 2**30
        reserved = batch.torch.cuda.memory_reserved() / 2**30
        memory = f" cuda={allocated:.1f}/{reserved:.1f}GiB"
    print(f"[{phase}] {completed}/{total} pairs elapsed={time.perf_counter() - started:.1f}s{memory}", flush=True)


def evaluate(model, tokenizer, records, *, suffix: str, batch_size: int, max_new_tokens: int,
             reuse_engine: str, label: str, progress_every: int, empty_cache_every: int,
             prefix_prompt_builder: Callable[[str], str] | None = None):
    rows, elapsed = [], 0.0
    phases = {} if reuse_engine == "scatter" else None
    phase_started = time.perf_counter()
    print(f"[{label}] start: pairs={len(records)} batch={batch_size} suffix_chars={len(suffix)}", flush=True)
    with raw_batch_adapter(batch, suffix, prefix_prompt_builder=prefix_prompt_builder), batch.torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            batch.torch.cuda.synchronize()
            began = time.perf_counter()
            directions = [(record, side) for record in chunk for side in ("a", "b")]
            if reuse_engine == "reference":
                values = batch.reuse_rows(model, tokenizer, directions, "reuse", max_new_tokens)
            else:
                values = reuse_rows_vectorized(batch, model, tokenizer, directions, max_new_tokens,
                                                assembly="scatter", phase_timing=phases)
            batch.torch.cuda.synchronize()
            elapsed += time.perf_counter() - began
            for index, record in enumerate(chunk):
                for offset, side in enumerate(("a", "b")):
                    value = values[2 * index + offset]
                    prediction = value["prediction"] or ""
                    if value["correct"] != score_response(record, side, prediction):
                        raise AssertionError("scoring mismatch")
                    rows.append({"task_id": record["task_id"], "attack_type": record["attack_type"],
                                 "target": side, "source_gold": record["gold_b" if side == "a" else "gold_a"],
                                 "reuse": value})
            del values
            batch_number = start // batch_size + 1
            if empty_cache_every and batch_number % empty_cache_every == 0:
                release_cuda()
            if batch_number % progress_every == 0 or start + len(chunk) == len(records):
                _progress(label, start + len(chunk), len(records), phase_started)
    timing = {"wall_seconds": elapsed}
    if phases is not None:
        timing.update({"donor_prefill_seconds": phases["donor_prefill"],
                       "cache_hit_seconds": sum(phases[name] for name in (
                           "target_prefix_prefill", "relocate_and_splice", "suffix_prefill", "decode"
                       )), "cache_hit_phases_seconds": phases})
    return rows, timing


def full_once(model, tokenizer, records, *, batch_size: int, max_new_tokens: int,
              progress_every: int, empty_cache_every: int):
    rows, elapsed = [], 0.0
    phase_started = time.perf_counter()
    print(f"[full] start: pairs={len(records)} batch={batch_size}", flush=True)
    with raw_batch_adapter(batch), batch.torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            batch.torch.cuda.synchronize()
            began = time.perf_counter()
            values, _ = batch.full_rows(model, tokenizer, chunk, "full", max_new_tokens)
            batch.torch.cuda.synchronize()
            elapsed += time.perf_counter() - began
            for index, record in enumerate(chunk):
                for offset, side in enumerate(("a", "b")):
                    value = values[2 * index + offset]
                    prediction = value["prediction"] or ""
                    if value["correct"] != score_response(record, side, prediction):
                        raise AssertionError("scoring mismatch")
                    rows.append({"task_id": record["task_id"], "attack_type": record["attack_type"],
                                 "target": side, "source_gold": record["gold_b" if side == "a" else "gold_a"],
                                 "full": value})
            del values
            batch_number = start // batch_size + 1
            if empty_cache_every and batch_number % empty_cache_every == 0:
                release_cuda()
            if batch_number % progress_every == 0 or start + len(chunk) == len(records):
                _progress("full", start + len(chunk), len(records), phase_started)
    return rows, elapsed


def join(full_rows, reuse_rows):
    full = {(row["task_id"], row["target"]): row for row in full_rows}
    reuse = {(row["task_id"], row["target"]): row for row in reuse_rows}
    if full.keys() != reuse.keys():
        raise ValueError("full/reuse evaluation keys differ")
    return [{**full[key], "reuse": reuse[key]["reuse"]} for key in full]


def release_cuda():
    """Prevent allocator-reserved KV buffers from accumulating across suffixes."""
    gc.collect()
    if batch.torch.cuda.is_available():
        batch.torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(batch.base.MODEL_IDS), default="1.7b")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=1,
                        help="print progress every N batches; 1 prints every batch")
    parser.add_argument("--empty-cache-every", type=int, default=1,
                        help="release CUDA allocator cache every N batches; 0 disables")
    parser.add_argument("--reuse-engine", choices=("scatter", "reference"), default="scatter",
                        help="scatter is the optimized batch KV assembly; reference is for comparison only")
    parser.add_argument("--run-dir", type=Path, help="write immutable per-step manifest here")
    parser.add_argument("--step", type=int, help="positive environment step number; required with --run-dir")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.progress_every < 1 or args.empty_cache_every < 0:
        parser.error("batch/token/progress values must be positive; empty-cache-every may be zero")
    if (args.run_dir is None) != (args.step is None):
        parser.error("--run-dir and --step must be provided together")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    candidates = parse_candidates(json.loads(args.candidates.read_text(encoding="utf-8")))
    records = batch.base.load_jsonl(args.input)
    for record in records:
        validate_generated_record(record)
    model_path = Path(batch.snapshot_download(batch.base.MODEL_IDS[args.model], local_files_only=True))
    tokenizer = batch.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = batch.AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=batch.torch.bfloat16,
                                                       device_map="cuda:0", attn_implementation="sdpa").eval()
    batch.base.assert_default_rope(model)
    full, full_seconds = full_once(model, tokenizer, records, batch_size=args.batch_size,
                                   max_new_tokens=args.max_new_tokens, progress_every=args.progress_every,
                                   empty_cache_every=args.empty_cache_every)
    release_cuda()
    baseline, baseline_timing = evaluate(model, tokenizer, records, suffix="",
                                          batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                                          reuse_engine=args.reuse_engine, label="direct_reuse",
                                          progress_every=args.progress_every, empty_cache_every=args.empty_cache_every)
    baseline_joined = join(full, baseline)
    del baseline
    release_cuda()
    report = {"input": str(args.input), "candidate_file": str(args.candidates), "model": args.model,
              "full_seconds": full_seconds, "full": {"summary": summarize(full), "results": full},
              "direct_reuse": {"timing_seconds": baseline_timing, "summary": summarize(baseline_joined),
                                 "results": baseline_joined}, "candidates": []}
    for candidate in candidates:
        reuse, timing = evaluate(model, tokenizer, records, suffix=candidate.suffix,
                                  batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                                  reuse_engine=args.reuse_engine, label=f"suffix:{candidate.candidate_id}",
                                  progress_every=args.progress_every, empty_cache_every=args.empty_cache_every)
        joined = join(full, reuse)
        report["candidates"].append({"candidate": candidate.__dict__, "timing_seconds": timing,
                                     "summary": summarize(joined), "results": joined})
        print(f"{candidate.candidate_id}: {json.dumps({'accuracy': report['candidates'][-1]['summary']['overall']['reuse_accuracy'], **timing})}", flush=True)
        # An individual candidate result is complete at this point. Persist it
        # before releasing its large temporary KV buffers; the final manifest
        # is still only written after all candidates complete.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**report, "complete": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        del reuse, joined
        release_cuda()
    report["reuse_engine"] = args.reuse_engine
    print(json.dumps({"full_seconds": full_seconds, "direct_reuse_timing_seconds": baseline_timing}, indent=2), flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.run_dir:
        manifest = write_step(args.run_dir, args.step, "defense_evaluation", {
            "attack_candidates": file_ref(args.input), "suffix_candidates": file_ref(args.candidates),
            "evaluation": file_ref(args.output), "model": args.model, "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens, "reuse_engine": args.reuse_engine,
            "full_seconds": full_seconds, "direct_reuse_timing_seconds": baseline_timing,
            "suffix_results": [{"candidate_id": item["candidate"]["candidate_id"],
                                "suffix": item["candidate"]["suffix"],
                                "timing_seconds": item["timing_seconds"],
                                "summary": item["summary"]} for item in report["candidates"]],
        })
        print(f"step_manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())