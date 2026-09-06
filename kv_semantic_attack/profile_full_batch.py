#!/usr/bin/env python3
"""Compare full no-reasoning batch latency for two synthetic task slices."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import batch_eval as batch  # noqa: E402
from kv_semantic_attack.eval_synthetic_batch import raw_batch_adapter, without_shared_data_tags  # noqa: E402


def prepare_inputs(tokenizer, records):
    """Replicate batch_eval.full_rows prompt construction and left padding."""
    parts = []
    for record in records:
        for side in ("a", "b"):
            parts.append(batch.build_parts(tokenizer, record, side, "full"))
    device = batch.base.input_device(model)
    width = max(int(part.full_ids.numel()) for part in parts)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    inputs = torch.full((len(parts), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(parts), width), dtype=torch.long, device=device)
    for row, part in enumerate(parts):
        length = int(part.full_ids.numel())
        inputs[row, width - length:] = part.full_ids.to(device)
        mask[row, width - length:] = 1
    return inputs, mask, width, [int(part.full_ids.numel()) for part in parts]


def generate_once(inputs, mask, tokenizer, max_new_tokens):
    return model.generate(input_ids=inputs, attention_mask=mask, max_new_tokens=max_new_tokens,
                          do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                          eos_token_id=tokenizer.eos_token_id)


def timed_runs(inputs, mask, tokenizer, max_new_tokens, repeats):
    generate_once(inputs, mask, tokenizer, max_new_tokens)
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        sequences = generate_once(inputs, mask, tokenizer, max_new_tokens)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    token_counts = [int((row != tokenizer.pad_token_id).sum()) for row in sequences[:, inputs.shape[1]:]]
    return times, token_counts


def profile_runs(inputs, mask, tokenizer, max_new_tokens):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        generate_once(inputs, mask, tokenizer, max_new_tokens)
        torch.cuda.synchronize()
    events = profiler.key_averages()
    def device_time(event):
        # Torch renamed CUDA-specific aggregates to device aggregates.
        return getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))

    return [{"name": event.key, "cpu_ms": round(event.self_cpu_time_total / 1000, 3),
             "cuda_ms": round(device_time(event) / 1000, 3), "calls": event.count}
            for event in sorted(events, key=device_time, reverse=True)[:20]]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=ROOT / "kv_semantic_attack/generated_tasks/all.jsonl")
parser.add_argument("--output", type=Path, default=ROOT / "kv_semantic_attack/eval_outputs/full_batch_profile.json")
parser.add_argument("--model", default="1.7b", choices=tuple(batch.base.MODEL_IDS))
parser.add_argument("--fast-start", type=int, default=384)
parser.add_argument("--slow-start", type=int, default=400)
parser.add_argument("--pairs", type=int, default=16)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--max-new-tokens", type=int, default=128)
args = parser.parse_args()
if args.pairs < 1 or args.repeats < 1:
    parser.error("--pairs and --repeats must be positive")
records = [without_shared_data_tags(json.loads(line)) for line in args.input.read_text().splitlines()]
slices = {"fast": records[args.fast_start:args.fast_start + args.pairs],
          "slow": records[args.slow_start:args.slow_start + args.pairs]}
if any(len(value) != args.pairs for value in slices.values()):
    parser.error("requested slice exceeds input")
path = Path(batch.snapshot_download(batch.base.MODEL_IDS[args.model], local_files_only=True))
tokenizer = batch.AutoTokenizer.from_pretrained(path, local_files_only=True)
model = batch.AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16,
                                                   device_map="cuda:0", attn_implementation="sdpa").eval()
batch.base.assert_default_rope(model)
report = {"model": args.model, "batch_pairs": args.pairs, "directions": 2 * args.pairs,
          "repeats": args.repeats, "max_new_tokens": args.max_new_tokens, "slices": {}}
with raw_batch_adapter(batch), torch.inference_mode():
    for label, records in slices.items():
        construction_started = time.perf_counter()
        inputs, mask, width, lengths = prepare_inputs(tokenizer, records)
        torch.cuda.synchronize()
        construction_seconds = time.perf_counter() - construction_started
        times, output_tokens = timed_runs(inputs, mask, tokenizer, args.max_new_tokens, args.repeats)
        operators = profile_runs(inputs, mask, tokenizer, args.max_new_tokens)
        report["slices"][label] = {
            "task_types": sorted({record["attack_type"] for record in records}),
            "prompt_build_and_pad_seconds": construction_seconds,
            "prompt_tokens": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths),
                              "padded_token_slots": width * len(lengths)},
            "generate_seconds": times, "output_nonpad_tokens": output_tokens,
            "top_cuda_operators": operators,
        }
        print(label, json.dumps(report["slices"][label], ensure_ascii=False), flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(args.output)