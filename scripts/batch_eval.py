#!/usr/bin/env python3
"""Batch smoke/evaluation runner for the six direct-reuse interfaces.

This file is intentionally standalone: it imports the existing direct-reuse
primitives but does not modify the official runner.  Full rows use native
Transformers ``generate``.  Reuse rows assemble their heterogeneous KV first,
then use one padded batched decoding loop.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_direct_reuse as base  # noqa: E402


METHODS = ("full", "reuse", "clean_reuse", "tail16_recompute",
           "tail16_post_recompute", "ours_post")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(base.MODEL_IDS), default="1.7b")
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl")
    parser.add_argument("--method", choices=METHODS, default="full")
    parser.add_argument("--limit", type=int, default=110)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--verify-limit", type=int, default=4,
                        help="also run batch_size=1 comparisons for the first N rows; 0 disables")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("batch_eval_results.json"))
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def pad_cache(rows, max_length):
    # Each item is ``(layers, true_length)``; the layer count belongs to the
    # first element, not to the two-element row wrapper.
    if not rows or not rows[0][0]:
        raise ValueError("cannot pad an empty KV cache")
    padded = []
    for layer in range(len(rows[0][0])):
        keys, values = [], []
        for cache, length in rows:
            key, value = cache[layer]
            pad = max_length - length
            keys.append(torch.nn.functional.pad(key, (0, 0, pad, 0)))
            values.append(torch.nn.functional.pad(value, (0, 0, pad, 0)))
        padded.append((torch.cat(keys, 0), torch.cat(values, 0)))
    return padded


def make_batch_cache(model, layers):
    """Construct a cache directly from already-batched KV tensors.

    Passing the tensors as ``ddp_cache_data`` is important here.  Creating an
    empty config-shaped cache and then updating it layer by layer can be
    interpreted differently by recent Transformers versions when the cache
    already contains a batch and left padding.
    """
    # Build the model-configured layer list first.  This guarantees that a
    # model with an explicit layer layout (Qwen3 has 28 layers) gets exactly
    # one cache layer per decoder layer.  Then install the already-computed
    # tensors directly; calling ``update`` here would append them to an empty
    # layer and is also incompatible with some cache implementations.
    try:
        cache = base.DynamicCache(config=model.config)
        if len(cache.layers) != len(layers):
            raise RuntimeError(
                f"cache layer mismatch: model has {len(cache.layers)}, KV has {len(layers)}"
            )
        for cache_layer, (key, value) in zip(cache.layers, layers):
            cache_layer.keys = key
            cache_layer.values = value
            cache_layer.dtype = key.dtype
            cache_layer.device = key.device
            cache_layer.is_initialized = True
    except (AttributeError, TypeError):
        # Compatibility with older Transformers releases.
        cache = base.DynamicCache()
        for layer_index, (key, value) in enumerate(layers):
            cache.update(key, value, layer_index)
    expected = layers[0][0].shape[-2]
    actual = cache.get_seq_length()
    if actual != expected:
        raise RuntimeError(f"batch cache length mismatch: expected {expected}, got {actual}")
    return cache


@torch.inference_mode()
def prefill_ids_batch(model, ids_list):
    """Prefill variable-length token sequences in one left-padded batch."""
    if not ids_list:
        return [], []
    device = base.input_device(model)
    lengths = [int(ids.numel()) for ids in ids_list]
    width = max(lengths)
    pad_id = 0
    input_ids = torch.full((len(ids_list), width), pad_id,
                           dtype=torch.long, device=device)
    attention = torch.zeros((len(ids_list), width), dtype=torch.long, device=device)
    positions = torch.zeros((len(ids_list), width), dtype=torch.long, device=device)
    for row, ids in enumerate(ids_list):
        n = lengths[row]
        input_ids[row, width - n:] = ids.to(device)
        attention[row, width - n:] = 1
        positions[row, width - n:] = torch.arange(n, device=device)
    output = model(
        input_ids=input_ids,
        attention_mask=attention,
        position_ids=positions,
        cache_position=torch.arange(width, device=device),
        use_cache=True,
        return_dict=True,
    )
    layers = base.extract_cache_tensors(output.past_key_values)
    logits = output.logits[
        torch.arange(len(ids_list), device=device),
        torch.tensor([width - 1] * len(ids_list), device=device),
    ].detach().float()
    trimmed = [
        [
            (key[row:row + 1, ..., width - lengths[row]:, :].contiguous(),
             value[row:row + 1, ..., width - lengths[row]:, :].contiguous())
            for key, value in layers
        ]
        for row in range(len(ids_list))
    ]
    return logits, trimmed


@torch.inference_mode()
def forward_suffix_batch(model, mixed_rows, suffix_ids):
    """Run all heterogeneous mixed-prefix suffixes in one prefill batch."""
    device = base.input_device(model)
    prefix_lengths = [int(layers[0][0].shape[-2]) for layers in mixed_rows]
    suffix_lengths = [int(ids.numel()) for ids in suffix_ids]
    max_prefix = max(prefix_lengths)
    width = max(suffix_lengths)
    padded_layers = pad_cache(
        list(zip(mixed_rows, prefix_lengths)), max_prefix
    )
    input_ids = torch.zeros((len(mixed_rows), width), dtype=torch.long, device=device)
    suffix_attention = torch.zeros_like(input_ids)
    positions = torch.zeros_like(input_ids)
    for row, ids in enumerate(suffix_ids):
        n = suffix_lengths[row]
        input_ids[row, :n] = ids.to(device)
        suffix_attention[row, :n] = 1
        positions[row, :n] = torch.arange(
            prefix_lengths[row], prefix_lengths[row] + n, device=device
        )
    attention = torch.zeros(
        (len(mixed_rows), max_prefix + width), dtype=torch.long, device=device
    )
    for row, length in enumerate(prefix_lengths):
        attention[row, max_prefix - length:max_prefix] = 1
        attention[row, max_prefix:] = suffix_attention[row]
    output = model(
        input_ids=input_ids,
        attention_mask=attention,
        position_ids=positions,
        cache_position=torch.arange(max_prefix, max_prefix + width, device=device),
        past_key_values=make_batch_cache(model, padded_layers),
        use_cache=True,
        return_dict=True,
    )
    row_ids = torch.arange(len(mixed_rows), device=device)
    last_ids = torch.tensor([n - 1 for n in suffix_lengths], device=device)
    logits = output.logits[row_ids, last_ids].detach().float()
    layers = base.extract_cache_tensors(output.past_key_values)
    trimmed = []
    for row, (prefix_length, suffix_length) in enumerate(
        zip(prefix_lengths, suffix_lengths)
    ):
        start = max_prefix - prefix_length
        end = start + prefix_length + suffix_length
        trimmed.append([
            (key[row:row + 1, ..., start:end, :].contiguous(),
             value[row:row + 1, ..., start:end, :].contiguous())
            for key, value in layers
        ])
    return logits, trimmed


@torch.inference_mode()
def decode_batch(model, tokenizer, logits, caches, lengths, max_new_tokens):
    device = base.input_device(model)
    max_length = max(lengths)
    batch = len(lengths)
    cache = make_batch_cache(model, caches)
    attention = torch.zeros((batch, max_length), dtype=torch.long, device=device)
    for row, length in enumerate(lengths):
        attention[row, max_length - length:] = 1
    current = logits.argmax(dim=-1).view(batch, 1).to(device)
    generated = [[int(current[row, 0])] for row in range(batch)]
    eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    done = torch.tensor([x[0] in eos_ids for x in generated], device=device)
    for step in range(1, max_new_tokens):
        if bool(done.all()):
            break
        output = model(
            input_ids=current,
            attention_mask=torch.cat(
                (attention, torch.ones((batch, step), dtype=torch.long, device=device)), dim=1
            ),
            position_ids=torch.tensor(
                [[lengths[row] + step - 1] for row in range(batch)],
                dtype=torch.long, device=device,
            ),
            cache_position=torch.tensor([max_length + step - 1], device=device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        # DynamicCache is mutated by the forward pass, but retaining the
        # returned object also supports implementations that return a new
        # cache wrapper for the next decoding step.
        cache = output.past_key_values
        nxt = output.logits[:, -1].argmax(dim=-1)
        if eos_ids:
            nxt = torch.where(done, torch.tensor(eos, device=device), nxt)
        for row in range(batch):
            generated[row].append(int(nxt[row]))
        done |= torch.tensor([int(x) in eos_ids for x in nxt], device=device)
        current = nxt.view(batch, 1)
    return generated


def build_parts(tokenizer, record, side, method):
    modified = dict(record)
    if method in {"tail16_post_recompute", "ours_post"}:
        modified = base.with_post_task_restatement(record, side)
    return base.build_prompt_parts(
        tokenizer, modified, side, enable_thinking=False,
        explicit_reasoning=False, boxed_output=True,
    )


def reuse_rows(model, tokenizer, rows, method, max_new_tokens):
    work = []
    recompute_indices = []
    for record, side in rows:
        source = "a" if side == "b" else "b"
        parts = {x: build_parts(tokenizer, record, x, method) for x in ("a", "b")}
        if not torch.equal(parts["a"].block_ids, parts["b"].block_ids):
            raise ValueError(f"{record['task_id']}: block tokenization mismatch")
        work.append((record, side, parts[side], parts[source], None, None))

    # Batch all ordinary prefills used by the reuse variants.  The KV
    # relocation itself remains row-specific, as it must use each row's token
    # offsets, but the expensive model prefills are shared by this batch.
    _, source_layers = prefill_ids_batch(model, [item[3].full_ids for item in work])
    _, target_layers = prefill_ids_batch(model, [item[2].prefix_ids for item in work])
    clean_layers = None
    if method == "clean_reuse":
        _, clean_layers = prefill_ids_batch(
            model, [item[2].block_ids for item in work]
        )
    work = [
        (item[0], item[1], item[2], item[3], source_layers[row], target_layers[row])
        for row, item in enumerate(work)
    ]
    assembled = []
    for row, (record, side, target_parts, source_parts, source_row, target_row) in enumerate(work):
        source = "a" if side == "b" else "b"
        parts = {"a": target_parts if side == "a" else source_parts,
                 "b": source_parts if side == "a" else target_parts}
        source_layers_row = source_row
        target_prefix = target_row
        clean = None
        if method == "clean_reuse":
            clean = base.slice_cache(clean_layers[row], 0, parts["a"].block_ids.numel())
        source_start = 0 if clean is not None else parts[source].block_token_start
        source_block = clean if clean is not None else base.slice_cache(
            source_layers_row, parts[source].block_token_start, parts[source].block_token_end
        )
        relocated = base.relocate_block(
            model, source_block, source_start, parts[side].block_token_start
        )
        block_len = parts[side].block_ids.numel()
        recompute_tail = method in {"tail16_recompute", "tail16_post_recompute"}
        if recompute_tail:
            donor_len = max(0, block_len - min(16, block_len))
            mixed = base.splice_prefix_block(
                target_prefix, base.slice_cache(relocated, 0, donor_len)
            )
        else:
            mixed = base.splice_prefix_block(target_prefix, relocated)
        assembled.append((record, side, parts[side], mixed, source_layers_row, target_prefix))
        if recompute_tail:
            recompute_indices.append(len(assembled) - 1)

    work = assembled

    # Recompute each row's complete tail in one heterogeneous causal prefill.
    # ``forward_suffix_batch`` left-pads the variable-length donor caches,
    # supplies row-specific position IDs, and unpads the updated caches.
    if recompute_indices:
        tail_caches = [work[index][3] for index in recompute_indices]
        tail_ids = []
        for index in recompute_indices:
            parts = work[index][2]
            block_len = int(parts.block_ids.numel())
            donor_len = max(0, block_len - min(16, block_len))
            tail_ids.append(parts.block_ids[donor_len:block_len])
        _, updated = forward_suffix_batch(model, tail_caches, tail_ids)
        for index, layers in zip(recompute_indices, updated):
            record, side, parts, _, source_layers, target_prefix = work[index]
            work[index] = (record, side, parts, layers, source_layers, target_prefix)

    mixed_rows = [item[3] for item in work]
    suffix_ids = [item[2].suffix_ids for item in work]
    suffix_logits, suffix_caches = forward_suffix_batch(model, mixed_rows, suffix_ids)
    prepared = []
    for row, (record, side, parts, mixed, source_layers, target_prefix) in enumerate(work):
        logits = suffix_logits[row]
        final_cache = suffix_caches[row]
        prepared.append((record, side, parts, logits, final_cache,
                         int(final_cache[0][0].shape[-2]), source_layers, target_prefix))
    if not prepared:
        return []
    caches = [(x[4], x[5]) for x in prepared]
    padded = pad_cache(caches, max(x[5] for x in prepared))
    logits = torch.cat([x[3].view(1, -1) for x in prepared], 0)
    lengths = [x[5] for x in prepared]
    generated = decode_batch(model, tokenizer, logits, padded, lengths, max_new_tokens)
    output = []
    for item, tokens in zip(prepared, generated):
        record, side, parts = item[:3]
        result = base.result_from_generation(
            tokenizer, record["dataset"], record[f"gold_{side}"], tokens,
            len(tokens) >= max_new_tokens, 0.0, max_new_tokens, False, False,
            "The answer is: \\boxed{",
        )
        output.append(result)
    return output


def full_rows(model, tokenizer, records, method, max_new_tokens):
    requests, metadata = [], []
    for record in records:
        for side in ("a", "b"):
            parts = build_parts(tokenizer, record, side, method)
            requests.append(parts)
            metadata.append((record, side, parts))
    device = base.input_device(model)
    width = max(int(x.full_ids.numel()) for x in requests)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    inputs = torch.full((len(requests), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(requests), width), dtype=torch.long, device=device)
    for row, parts in enumerate(requests):
        n = int(parts.full_ids.numel())
        inputs[row, width - n:] = parts.full_ids.to(device)
        mask[row, width - n:] = 1
    started = time.perf_counter()
    sequences = model.generate(
        input_ids=inputs, attention_mask=mask, max_new_tokens=max_new_tokens,
        do_sample=False, use_cache=True, pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - started
    output = []
    for row, (record, side, _) in enumerate(metadata):
        tokens = sequences[row, width:].tolist()
        output.append(base.result_from_generation(
            tokenizer, record["dataset"], record[f"gold_{side}"], tokens,
            len(tokens) >= max_new_tokens, elapsed, max_new_tokens, False, False,
            "The answer is: \\boxed{",
        ))
    return output, elapsed


def main() -> int:
    args = parse_args()
    records = base.load_jsonl(args.input)
    records = [x for x in records if x.get("dataset") == "argkp"][:args.limit]
    if not records:
        raise ValueError("no argkp records selected")
    model_path = Path(snapshot_download(
        base.MODEL_IDS[args.model], local_files_only=not args.allow_download
    ))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa",
    )
    model.eval()
    base.assert_default_rope(model)
    results = []
    verification = []
    batch_wall_seconds = 0.0
    for start in range(0, len(records), args.batch_size):
        chunk = records[start:start + args.batch_size]
        started = time.perf_counter()
        if args.method == "full":
            values, _ = full_rows(model, tokenizer, chunk, args.method, args.max_new_tokens)
        else:
            values = reuse_rows(
                model, tokenizer,
                [(record, side) for record in chunk for side in ("a", "b")],
                args.method, args.max_new_tokens,
            )
        elapsed = time.perf_counter() - started
        batch_wall_seconds += elapsed
        results.extend(values)
        print(f"[{start + len(chunk)}/{len(records)}] method={args.method} "
              f"batch_size={len(chunk)} elapsed={elapsed:.3f}s", flush=True)
        verify_count = max(0, min(len(chunk), args.verify_limit - start))
        for local_index in range(verify_count):
            record = chunk[local_index]
            if args.method == "full":
                single, single_elapsed = full_rows(
                    model, tokenizer, [record], args.method, args.max_new_tokens
                )
            else:
                single_started = time.perf_counter()
                single = reuse_rows(
                    model, tokenizer, [(record, "a"), (record, "b")],
                    args.method, args.max_new_tokens,
                )
                single_elapsed = time.perf_counter() - single_started
            batch_slice = values[local_index * 2:local_index * 2 + 2]
            for side_index, (batch_value, single_value) in enumerate(zip(batch_slice, single)):
                verification.append({
                    "task_id": record["task_id"],
                    "side": "a" if side_index == 0 else "b",
                    "batch_prediction": batch_value["prediction"],
                    "single_prediction": single_value["prediction"],
                    "prediction_match": batch_value["prediction"] == single_value["prediction"],
                    "common_prefix_chars": next((i for i, (x, y) in enumerate(
                        zip(batch_value["output_text"], single_value["output_text"])) if x != y),
                        min(len(batch_value["output_text"]), len(single_value["output_text"]))),
                    "batch_output_tokens": batch_value["output_tokens"],
                    "single_output_tokens": single_value["output_tokens"],
                    "single_elapsed_seconds": single_elapsed,
                })

    summary = {
        "method": args.method, "model": args.model, "records": len(records),
        "batch_size": args.batch_size,
        "elapsed_seconds": batch_wall_seconds,
        "accuracy": sum(x["correct"] for x in results) / len(results),
        "parse_rate": sum(x["prediction"] is not None for x in results) / len(results),
        "verification": verification,
        "results": results,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("method", "records", "batch_size", "accuracy", "parse_rate")}, indent=2))
    if verification:
        print(json.dumps({
            "verified_rows": len({x["task_id"] for x in verification}),
            "verified_directions": len(verification),
            "prediction_matches": sum(x["prediction_match"] for x in verification),
            "mean_common_prefix_chars": sum(x["common_prefix_chars"] for x in verification) / len(verification),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
