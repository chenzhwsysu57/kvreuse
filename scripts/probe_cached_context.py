#!/usr/bin/env python3
"""Probe whether Qwen can infer which text was supplied through KV reuse.

The model is never given an explicit cache label. The only difference between
full and reuse conditions is how the shared block KV is produced and attached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# When this file is executed as python scripts/probe_cached_context.py,
# Python puts scripts/ on sys.path rather than the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_direct_reuse import (
    MODEL_IDS,
    assert_default_rope,
    build_prompt_parts,
    forward_ids,
    forward_suffix,
    greedy_continue,
    relocate_block,
    slice_cache,
    splice_prefix_block,
)


PREFIX_A = (
    "You are walking through a quiet library in the late afternoon. "
    "Tall shelves hold old books, dust floats in golden window light, and "
    "a librarian is arranging returned novels near the reading room."
)
PREFIX_B = (
    "You are standing beside the sea at sunset. "
    "The horizon is orange and violet, waves are folding onto the shore, "
    "and a cool wind is carrying the smell of salt."
)
SHARED_BLOCK = (
    "Xiaohong is making a phone call. She checks the time, listens carefully, "
    "asks one short question, and writes the important part of the conversation "
    "in a small notebook."
)
QUESTIONS = {
    "repeat": (
        "Please reproduce, word for word, all text that appeared earlier in this "
        "user message, including the preceding context and shared block. Do not "
        "summarize or explain; output the complete earlier text."
    ),
    "identify_cached": (
        "Which part of the preceding context is the cached context? Identify the "
        "specific text and explain briefly how you know it is cached."
    ),
}


def make_record(prefix: str, question: str) -> dict[str, str]:
    return {
        "prefix_a": prefix,
        "prefix_b": prefix,
        "shared_block": SHARED_BLOCK,
        "question": question,
    }


@torch.inference_mode()
def generate_full(model: Any, tokenizer: Any, parts: Any, max_new_tokens: int) -> tuple[str, bool]:
    logits, layers = forward_ids(model, parts.full_ids)
    tokens, hit_limit = greedy_continue(model, tokenizer, logits, layers, max_new_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True).strip(), hit_limit


@torch.inference_mode()
def generate_reuse(
    model: Any, tokenizer: Any, donor: Any, target: Any, max_new_tokens: int
) -> tuple[str, bool]:
    if not torch.equal(donor.block_ids, target.block_ids):
        raise ValueError("donor and target shared block tokenization differs")
    _, donor_layers = forward_ids(model, donor.full_ids)
    _, target_prefix_layers = forward_ids(model, target.prefix_ids)
    donor_block = slice_cache(donor_layers, donor.block_token_start, donor.block_token_end)
    relocated = relocate_block(
        model, donor_block, donor.block_token_start, target.block_token_start
    )
    mixed = splice_prefix_block(target_prefix_layers, relocated)
    logits, prompt_layers = forward_suffix(model, mixed, target.suffix_ids)
    tokens, hit_limit = greedy_continue(
        model, tokenizer, logits, prompt_layers, max_new_tokens
    )
    return tokenizer.decode(tokens, skip_special_tokens=True).strip(), hit_limit


def run_probe(model: Any, tokenizer: Any, question: str, max_new_tokens: int) -> dict[str, Any]:
    records = {"a": make_record(PREFIX_A, question), "b": make_record(PREFIX_B, question)}
    parts = {
        side: build_prompt_parts(
            tokenizer, record, side, enable_thinking=False,
            explicit_reasoning=False, boxed_output=False,
        )
        for side, record in records.items()
    }
    full_a, full_a_limit = generate_full(model, tokenizer, parts["a"], max_new_tokens)
    full_b, full_b_limit = generate_full(model, tokenizer, parts["b"], max_new_tokens)
    reuse_b_to_a, reuse_b_to_a_limit = generate_reuse(
        model, tokenizer, parts["b"], parts["a"], max_new_tokens
    )
    reuse_a_to_b, reuse_a_to_b_limit = generate_reuse(
        model, tokenizer, parts["a"], parts["b"], max_new_tokens
    )
    return {
        "question": question,
        "inputs": records,
        "conditions": {
            "full_a": {"output": full_a, "hit_max_new_tokens": full_a_limit},
            "full_b": {"output": full_b, "hit_max_new_tokens": full_b_limit},
            "reuse_b_to_a": {
                "donor": "b", "target": "a", "output": reuse_b_to_a,
                "hit_max_new_tokens": reuse_b_to_a_limit,
            },
            "reuse_a_to_b": {
                "donor": "a", "target": "b", "output": reuse_a_to_b,
                "hit_max_new_tokens": reuse_a_to_b_limit,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=("4b",), default="4b")
    parser.add_argument("--output-dir", default="./cached_context_probe")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(snapshot_download(
        MODEL_IDS[args.model_size], local_files_only=not args.allow_download
    ))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa",
    )
    model.eval()
    assert_default_rope(model)

    result = {
        "model": str(model_path),
        "prefix_a": PREFIX_A,
        "prefix_b": PREFIX_B,
        "shared_block": SHARED_BLOCK,
        "probes": {
            name: run_probe(model, tokenizer, question, args.max_new_tokens)
            for name, question in QUESTIONS.items()
        },
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
