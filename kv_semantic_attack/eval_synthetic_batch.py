"""Full/direct-reuse evaluation using scripts.batch_eval's batch primitives.

Only prompt construction and result formatting are adapted, temporarily within
a scoped context. It uses the repository's no-reasoning contract: an unfinished
assistant ``The answer is: \\boxed{`` prefix and a closing boxed answer.
No corrective text or token recomputation is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kv_semantic_attack.synthetic_tasks import score_response, validate_generated_record


def _extract_boxed_answer(text):
    """Return the balanced content of the final ``\\boxed{...}`` expression."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index, char in enumerate(text[content_start:], content_start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]
    tail = text[content_start:].strip()
    return tail if len(tail) == 1 and tail in "ABCDEFGH" else None


def raw_result(tokenizer, dataset, gold, token_ids, hit_limit, elapsed,
               generation_limit, enable_thinking, explicit_reasoning=False,
               output_prefix=""):
    """Decode the standard no-reasoning boxed response, preserving answer bytes."""
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    trimmed = []
    stopped = False
    for token in token_ids:
        trimmed.append(token)
        if token in eos_ids:
            stopped = True
            break
        generated = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
        text = f"{output_prefix}{generated}" if output_prefix else generated
        answer = _extract_boxed_answer(text)
        return {"output_text": text, "answer_text": answer, "prediction": answer, "gold": gold,
            "correct": answer == gold, "output_token_ids": trimmed,
            "output_tokens": len(trimmed),
            "hit_max_new_tokens": not stopped and len(trimmed) >= generation_limit}


@contextmanager
def raw_batch_adapter(batch, suffix: str = ""):
    def build_parts(tokenizer, record, side, method):
        if method not in ("full", "reuse"):
            raise ValueError("this evaluation permits only uncorrected full/reuse")
        modified = dict(record)
        if suffix:
            # `question` is after the shared block in build_prompt_parts, so
            # this leaves donor KV unchanged and is precisely a post-block suffix.
            modified["question"] = suffix + "\n\n" + record["question"]
        return batch.base.build_prompt_parts(
            tokenizer, modified, side, enable_thinking=False,
            explicit_reasoning=False, boxed_output=True,
        )

    with patch.object(batch, "build_parts", build_parts), patch.object(
        batch.base, "result_from_generation", raw_result
    ):
        yield


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["attack_type"]].append(row)
    output = {}
    for name, values in [("overall", rows), *sorted(groups.items())]:
        if not values:
            continue
        full_rows = [row["full"] for row in values if "full" in row]
        reuse_rows = [row["reuse"] for row in values if "reuse" in row]
        full_correct = sum(row["correct"] for row in full_rows)
        reuse_correct = sum(row["correct"] for row in reuse_rows)
        paired = [row for row in values if "full" in row and "reuse" in row]
        regressions = sum(row["full"]["correct"] and not row["reuse"]["correct"] for row in paired)
        paired_full_correct = sum(row["full"]["correct"] for row in paired)
        output[name] = {
            "directions": len(values),
            "full_directions": len(full_rows), "reuse_directions": len(reuse_rows),
            "full_correct": full_correct if full_rows else None,
            "reuse_correct": reuse_correct if reuse_rows else None,
            "full_accuracy": full_correct / len(full_rows) if full_rows else None,
            "reuse_accuracy": reuse_correct / len(reuse_rows) if reuse_rows else None,
            "full_minus_reuse_pp": 100 * sum(int(row["full"]["correct"]) - int(row["reuse"]["correct"]) for row in paired) / len(paired) if paired else None,
            "reuse_failures_on_full_correct": regressions if paired else None,
            "reuse_failure_rate_on_full_correct": regressions / paired_full_correct if paired_full_correct else None,
            "source_gold_match_rate": sum(row["reuse"]["prediction"] == row["source_gold"] for row in values if "reuse" in row) / len(reuse_rows) if reuse_rows else None,
            "full_truncated": sum(row["hit_max_new_tokens"] for row in full_rows) if full_rows else None,
            "reuse_truncated": sum(row["hit_max_new_tokens"] for row in reuse_rows) if reuse_rows else None,
        }
    return output


def save_report(output, config, results, verification, timing, complete):
    report = {"complete": complete, "config": config, "timing_seconds": timing,
              "verification": verification, "summary": summarize(results), "results": results}
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def without_shared_data_tags(record):
    """Remove legacy boundary tags when present; current datasets omit them."""
    block = record["shared_block"]
    opening, closing = "<shared_data>", "</shared_data>"
    if not block.startswith(opening) and not block.endswith(closing):
        return dict(record)
    if not block.startswith(opening) or not block.endswith(closing):
        raise ValueError("shared_data boundary tags are incomplete")
    return {**record, "shared_block": block[len(opening):-len(closing)]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "kv_semantic_attack/generated_tasks/all.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=("0.6b", "1.7b", "4b", "8b"), default="1.7b")
    parser.add_argument("--method", choices=("full", "reuse", "both"), default="both")
    parser.add_argument("--suffix-file", type=Path,
                        help="UTF-8 post-block suffix; applied only to reuse, not Full")
    parser.add_argument("--reuse-engine", choices=("reference", "vectorized", "scatter"), default="reference",
                        help="reference is scripts/batch_eval.py; vectorized avoids per-direction KV clones")
    parser.add_argument("--strip-shared-data-tags", action="store_true",
                        help="remove boundary tags after validating input; retain internal data and whitespace")
    parser.add_argument("--batch-size", type=int, default=16, help="pairs per batch; twice as many directions")
    parser.add_argument("--per-type-limit", type=int, default=0, help="0 selects all")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--verify-per-type", type=int, default=0, help="extra serial/self-reuse checks; disabled for fast iteration")
    parser.add_argument("--save-every", type=int, default=0, help="checkpoint every N batches; 0 writes only the final report")
    parser.add_argument("--allow-batch-mismatch", action="store_true",
                        help="record batch/serial deviations without aborting; self-reuse checks still must pass")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.per_type_limit < 0 or args.verify_per_type < 0 or args.save_every < 0:
        parser.error("invalid batch size, token limit or per-type limits")
    if args.verify_per_type and args.method != "both":
        parser.error("--verify-per-type requires --method both; single-method runs do not execute the other method")
    if args.output is None:
        name = "full_reuse" if args.method == "both" else args.method
        if args.strip_shared_data_tags:
            name += "_no_tags"
        args.output = ROOT / f"kv_semantic_attack/eval_outputs/{name}_{args.model}_batch{args.batch_size}.json"
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}")
    return args


def evaluate_chunk(batch, model, tokenizer, chunk, methods, max_new_tokens, timing, reuse_engine,
                   reuse_phases=None):
    """Run exactly the requested methods; never perform hidden verification."""
    values = {}
    for method in methods:
        batch.torch.cuda.synchronize()
        began = time.perf_counter()
        if method == "full":
            values[method], _ = batch.full_rows(model, tokenizer, chunk, "full", max_new_tokens)
        elif method == "reuse":
            rows = [(record, side) for record in chunk for side in ("a", "b")]
            if reuse_engine == "reference":
                values[method] = batch.reuse_rows(model, tokenizer, rows, "reuse", max_new_tokens)
            else:
                from kv_semantic_attack.batched_reuse import reuse_rows_vectorized
                values[method] = reuse_rows_vectorized(
                    batch, model, tokenizer, rows, max_new_tokens,
                    assembly="scatter" if reuse_engine == "scatter" else "loop",
                    phase_timing=reuse_phases,
                )
        else:
            raise ValueError(f"unsupported method: {method}")
        batch.torch.cuda.synchronize()
        timing[method] += time.perf_counter() - began
    return values


def cuda_memory_gib():
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def main():
    args = parse_args()
    methods = ["full", "reuse"] if args.method == "both" else [args.method]
    suffix = args.suffix_file.read_text(encoding="utf-8").strip() if args.suffix_file else ""
    if args.suffix_file and not suffix:
        raise ValueError("--suffix-file is empty")
    started = time.perf_counter()
    import batch_eval as batch

    records = batch.base.load_jsonl(args.input)
    counts = defaultdict(int)
    selected = []
    for record in records:
        validate_generated_record(record)
        name = record["attack_type"]
        if args.per_type_limit and counts[name] >= args.per_type_limit:
            continue
        counts[name] += 1
        if args.strip_shared_data_tags:
            record = without_shared_data_tags(record)
        selected.append(record)
    if not selected:
        raise ValueError("no records selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {"input": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "model": args.model, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
              "counts": dict(counts), "reasoning": False, "boxed_output": True,
              "output_protocol": "repository_no_reasoning_boxed_assistant_prefill",
              "strip_shared_data_tags": args.strip_shared_data_tags,
              "effective_input_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              "allow_batch_mismatch": args.allow_batch_mismatch,
              "semantic_repair": False, "methods": methods,
              "post_block_suffix": suffix,
              "post_block_suffix_chars": len(suffix),
              "reuse_engine": args.reuse_engine,
              "verify_per_type": args.verify_per_type, "save_every": args.save_every,
              "note": "Reuse includes positional RoPE relocation, not semantic correction. Wall time includes donor construction, not serving speedup."}
    path = Path(batch.snapshot_download(batch.base.MODEL_IDS[args.model], local_files_only=True))
    tokenizer = batch.AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = batch.AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=batch.torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa",
    )
    model.eval()
    batch.base.assert_default_rope(model)
    config["model_path"] = str(path)
    config["torch_version"] = batch.torch.__version__
    results, verification = [], []
    verified = defaultdict(int)
    timing = {method: 0.0 for method in methods}
    reuse_phases = {} if args.method == "reuse" and args.reuse_engine != "reference" else None
    timing["setup"] = time.perf_counter() - started
    print(f"[ready] pairs={len(selected)} methods={','.join(methods)} batch={args.batch_size} "
          f"verify_per_type={args.verify_per_type} save_every={args.save_every}", flush=True)

    with raw_batch_adapter(batch, suffix if "reuse" in methods else ""), batch.torch.inference_mode():
        for start in range(0, len(selected), args.batch_size):
            chunk = selected[start:start + args.batch_size]
            batch.torch.cuda.reset_peak_memory_stats()
            chunk_started = time.perf_counter()
            values = evaluate_chunk(batch, model, tokenizer, chunk, methods, args.max_new_tokens, timing,
                                    args.reuse_engine, reuse_phases)
            # Used only by explicit paired verification below.
            full, reuse = values.get("full", []), values.get("reuse", [])
            for index, record in enumerate(chunk):
                name = record["attack_type"]
                for offset, side in enumerate(("a", "b")):
                    row = {"task_id": record["task_id"], "attack_type": name,
                           "target": side, "source_gold": record["gold_b" if side == "a" else "gold_a"]}
                    for method in methods:
                        value = values[method][2 * index + offset]
                        assert value["correct"] == score_response(record, side, value["prediction"] or "")
                        row[method] = value
                    results.append(row)
                if verified[name] < args.verify_per_type:
                    verified[name] += 1
                    # Each direction separately: truly batch-size-one references.
                    for offset, side in enumerate(("a", "b")):
                        control = dict(record)
                        for target in ("a", "b"):
                            control[f"prefix_{target}"] = record[f"prefix_{side}"]
                            control[f"gold_{target}"] = record[f"gold_{side}"]
                        parts = batch.build_parts(tokenizer, record, side, "full")
                        logits, cache = batch.base.forward_ids(model, parts.full_ids)
                        tokens, _ = batch.base.greedy_continue(model, tokenizer, logits, cache, args.max_new_tokens)
                        f = raw_result(tokenizer, record["dataset"], record[f"gold_{side}"], tokens,
                                       False, 0, args.max_new_tokens, False, output_prefix="The answer is: \\boxed{")
                        r = batch.reuse_rows(model, tokenizer, [(record, side)], "reuse", args.max_new_tokens)[0]
                        self_r = batch.reuse_rows(model, tokenizer, [(control, side)], "reuse", args.max_new_tokens)[0]
                        check = {"task_id": record["task_id"], "attack_type": name, "side": side,
                                 "full_batch_match": f["output_token_ids"] == full[2 * index + offset]["output_token_ids"],
                                 "reuse_batch_match": r["output_token_ids"] == reuse[2 * index + offset]["output_token_ids"],
                                 "self_reuse_matches_full": f["output_token_ids"] == self_r["output_token_ids"]}
                        verification.append(check)
                        if not all(check[key] for key in ("full_batch_match", "reuse_batch_match", "self_reuse_matches_full")):
                            check["outputs"] = {"serial_full": f, "serial_reuse": r, "self_reuse": self_r,
                                                "batch_full": full[2 * index + offset], "batch_reuse": reuse[2 * index + offset]}
                            save_report(args.output, config, results, verification, timing, False)
                            if not check["self_reuse_matches_full"] or not args.allow_batch_mismatch:
                                raise RuntimeError(f"consistency check failed: {name}/{side}; see saved report")
                            print(f"[verification warning] batch/serial output differs: {name}/{side}", flush=True)
            batch_number = start // args.batch_size + 1
            if args.save_every and batch_number % args.save_every == 0:
                save_report(args.output, config, results, verification, timing, False)
            # Report only this batch, not a rescan of every previous result.
            scores = " ".join(f"{method}={sum(x['correct'] for x in values[method])}/{len(values[method])}" for method in methods)
            memory = cuda_memory_gib()
            print(f"[{start + len(chunk)}/{len(selected)} pairs] {scores} elapsed={time.perf_counter() - chunk_started:.2f}s "
                f"peak={memory.get('allocated_gib', 0):.1f}/{memory.get('reserved_gib', 0):.1f}GiB", flush=True)
    timing["wall_before_final_save"] = time.perf_counter() - started
    if reuse_phases is not None:
        timing["cache_build_donor_prefill"] = reuse_phases["donor_prefill"]
        timing["cache_hit_latency"] = sum(
            reuse_phases[name] for name in ("target_prefix_prefill", "relocate_and_splice", "suffix_prefill", "decode")
        )
        timing["cache_hit_phases"] = reuse_phases
    save_report(args.output, config, results, verification, timing, True)
    print(json.dumps(summarize(results), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())