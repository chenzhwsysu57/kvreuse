#!/usr/bin/env python3
"""Run RelayCaching, CacheBlend, or EPIC on the project's A/B reuse tasks.

This is a *static-block adapter* for RelayCaching.  The upstream ``agent``
message is the already-written ``shared_block``: it is prefetched under the
source side's prefix, sliced from that real source-context cache, and then
provided to RelayCaching as ``{agent_<source>_current}`` in the target prompt.
Thus no text is regenerated merely to manufacture an upstream message.

Only the requested method is executed.  Existing full-prefill and direct
cross-reuse results remain the references; this runner deliberately does not
rerun them.

Run with the dedicated environment, for example:

  PYTHONPATH=third_party/RelayCaching /home/czw/miniconda3/envs/relaycaching/bin/python \
    scripts/run_relay_methods.py --method cacheblend --model 1.7b --limit 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
import uuid
from types import MethodType
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RELAY_ROOT = ROOT / "third_party" / "RelayCaching"
if str(RELAY_ROOT) not in sys.path:
    sys.path.insert(0, str(RELAY_ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from relay_deal_adapter import BOXED_PREFILL, SYSTEM_MESSAGE, iter_examples, load_jsonl, parse_prediction, write_metadata
from RelayCaching.llm.config import RelayCachingConfig
from RelayCaching.llm.gpt_chat import LLMChat


MODELSCOPE_CACHE = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
MODEL_PATHS = {
    "0.6b": str(MODELSCOPE_CACHE / "models/Qwen--Qwen3-0.6B/snapshots/master"),
    "1.7b": str(MODELSCOPE_CACHE / "models/Qwen--Qwen3-1.7B/snapshots/master"),
    "4b": str(MODELSCOPE_CACHE / "models/Qwen--Qwen3-4B/snapshots/master"),
    "8b": str(MODELSCOPE_CACHE / "models/Qwen--Qwen3-8B/snapshots/master"),
}
METHODS = ("relaycaching", "cacheblend", "epic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--model", default="1.7b", choices=sorted(MODEL_PATHS))
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/all.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/relay_methods")
    parser.add_argument("--limit", type=int, default=None, help="Limit original records, before two directions.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--explicit-reasoning", action="store_true",
        help="Ask for visible step-by-step reasoning followed by exactly one boxed answer."
    )
    parser.add_argument("--directions", nargs="+", choices=("a_to_b", "b_to_a"), default=("a_to_b", "b_to_a"))
    parser.add_argument(
        "--use-debug-selection-direction", action="store_true",
        help="For selector-produced debug JSONL, evaluate only each row's debug_selection.condition."
    )
    parser.add_argument("--cacheblend-diff-por", type=float, default=0.2)
    parser.add_argument("--epic-prefix-tokens", type=int, default=16)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--det-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=-1)
    parser.add_argument("--tau-dev", type=float, default=0.0)
    parser.add_argument("--tau-inf", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _install_project_qwen_template(chat: LLMChat, *, assistant_prefill: str | None) -> None:
    """Use the *same* Qwen3 rendering contract as ``run_direct_reuse.py``.

    RelayCaching normally appends a string after an assistant generation header.
    Our no-reasoning experiment instead has an unfinished assistant message
    (``The answer is: \\boxed{``), so that the first generated token attends to
    both the reused block and this suffix.  ``continue_final_message`` is the
    tokenizer-supported way to express that contract.
    """
    def build(self: LLMChat, messages: Any, assistant_prompt: str | None = None,
              add_generation_prompt: bool = True):
        normalised = self._normalise_messages(messages)
        if assistant_prefill is None:
            rendered = self.tokenizer.apply_chat_template(
                normalised, tokenize=False, add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        else:
            rendered = self.tokenizer.apply_chat_template(
                normalised + [{"role": "assistant", "content": assistant_prefill}],
                tokenize=False, add_generation_prompt=False,
                continue_final_message=True, enable_thinking=False,
            )
        ids = self.tokenizer.encode(rendered, return_tensors="pt", add_special_tokens=False)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        return inputs, rendered, int(ids.shape[-1])
    chat._build_chat_inputs = MethodType(build, chat)


def _slice_shared_block_cache(
    source: LLMChat,
    *,
    source_prefix: str,
    shared_block: str,
) -> tuple[Any, dict[str, torch.Tensor], dict[str, int]]:
    """Materialise block KV under the source prefix and convert it to relative RoPE.

    The block boundaries are computed from the model's own chat-template token
    streams, rather than by separately tokenizing text fragments.  This avoids
    silently changing Qwen's whitespace/BPE boundary tokens.
    """
    full_messages = source._render_base_messages(
        SYSTEM_MESSAGE, f"{source_prefix}\n\n{shared_block}"
    )
    full_inputs, rendered, _ = source._build_chat_inputs(full_messages, add_generation_prompt=True)
    user_text = f"{source_prefix}\n\n{shared_block}"
    user_start = rendered.find(user_text)
    if user_start < 0:
        raise RuntimeError("Qwen chat template did not preserve source user text verbatim")
    char_start = user_start + len(source_prefix) + 2
    char_end = char_start + len(shared_block)
    encoded = source.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    token_ids = encoded["input_ids"]
    if token_ids != full_inputs["input_ids"][0].tolist():
        raise RuntimeError("offset tokenizer and source prefill tokenizer disagree")
    # Same token-boundary rule as run_direct_reuse.py: retain an end-crossing
    # token with the reusable block, but keep a start-crossing token in prefix.
    nonempty = [(i, start, end) for i, (start, end) in enumerate(offsets) if end > start]
    starts = [i for i, start, _ in nonempty if start >= char_start]
    ends = [i for i, start, _ in nonempty if start >= char_end]
    if not starts:
        raise RuntimeError("shared block has no token-aligned start")
    block_start = starts[0]
    block_end = ends[0] if ends else len(token_ids)
    if not (0 <= block_start < block_end <= full_inputs["input_ids"].shape[-1]):
        raise RuntimeError(
            "Could not isolate shared_block in the Qwen chat template: "
            f"start={block_start}, end={block_end}, full={full_inputs['input_ids'].shape[-1]}"
        )
    with torch.no_grad():
        out = source.model(**full_inputs, use_cache=True, return_dict=True)
    cache = out.past_key_values.copy().slice_(start=block_start, end=block_end)
    # RelayCaching stores upstream response KV at relative positions, then its
    # official path rotates it to the target placeholder position.
    cache = source.anchor_engine.apply_rotary_pos_emb(cache, offset=-block_start)
    ids = {
        "input_ids": full_inputs["input_ids"][:, block_start:block_end],
        "attention_mask": torch.ones_like(full_inputs["input_ids"][:, block_start:block_end]),
    }
    return cache, ids, {
        "source_block_start": block_start, "source_block_end": block_end,
        "source_block_char_start": char_start, "source_block_char_end": char_end,
    }


def _store_static_agent_block(
    source: LLMChat,
    *,
    message: str,
    cache: Any,
    ids: dict[str, torch.Tensor],
) -> None:
    """Populate the exact shared-memory slots used by agent placeholders."""
    memory = source._shared_kv_cache_memory[source.node_id]
    for key, value in (
        ("response", cache),
        ("response_ids", ids),
        ("response_drop_num", 0),
        # RelayCaching's hidden-state path is optional.  Empty entries retain
        # the official fallback when a static upstream block has no generated
        # response hidden states to transmit.
        ("hidden_states", []),
        ("hidden_states_layer_idx", 0),
        ("additional_recompute_indices", []),
        ("additional_recompute_scores", []),
    ):
        memory.setdefault(key, {}).setdefault(message, []).append(value)


def _make_config(args: argparse.Namespace) -> RelayCachingConfig:
    return RelayCachingConfig(
        execution_mode=args.method,
        max_tokens=args.max_new_tokens,
        do_sample=False,
        start_layer=args.start_layer,
        det_layer=args.det_layer,
        end_layer=args.end_layer,
        tau_dev=args.tau_dev,
        tau_inf=args.tau_inf,
        cacheblend_diff_por=args.cacheblend_diff_por,
        epic_prefix_token_num=args.epic_prefix_tokens,
    ).validate()


async def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("RelayCaching evaluation requires a CUDA GPU; submit this command to a GPU node.")
    records = load_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No records loaded from {args.input}")
    if args.use_debug_selection_direction:
        examples = []
        for record in records:
            direction = record.get("debug_selection", {}).get("condition")
            if direction not in {"a_to_b", "b_to_a"}:
                raise ValueError(
                    "--use-debug-selection-direction requires every record to contain "
                    "debug_selection.condition equal to a_to_b or b_to_a"
                )
            examples.extend(iter_examples([record], directions=(direction,)))
    else:
        examples = list(iter_examples(records, directions=args.directions))
    # Keep visible-reasoning outputs separate from the established no-reasoning
    # baseline folders, even when callers reuse the same output root.
    output_base = args.output_root / "reasoning" if args.explicit_reasoning else args.output_root
    output_dir = output_base / args.method / f"qwen3-{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "samples.jsonl"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"{result_path} exists; use --overwrite or a new --output-root")
    metadata_path = output_dir / "adapter_prompts.jsonl"
    write_metadata(metadata_path, records, explicit_reasoning=args.explicit_reasoning)

    config = _make_config(args)
    model_path = MODEL_PATHS[args.model]
    # Two lightweight facades share one loaded model and RelayCaching's official
    # shared-memory store, while keeping source/target node identities separate.
    source = LLMChat(model_path, config=config)
    target = LLMChat(model_path, config=config)
    _install_project_qwen_template(source, assistant_prefill=None)
    _install_project_qwen_template(
        target, assistant_prefill=None if args.explicit_reasoning else BOXED_PREFILL
    )

    correct = 0
    completed = 0
    failures = 0
    with result_path.open("w", encoding="utf-8") as output:
        for index, example in enumerate(examples, start=1):
            message = f"{example.task_id}::{example.source}_to_{example.target}"
            request_uid = f"static-block-{uuid.uuid4().hex}"
            source.set_id(example.source, "source")
            target.set_id(example.target, "target")
            started = time.perf_counter()
            try:
                block_cache, block_ids, boundaries = _slice_shared_block_cache(
                    source,
                    source_prefix=example.source_prefix,
                    shared_block=example.shared_block,
                )
                _store_static_agent_block(source, message=message, cache=block_cache, ids=block_ids)
                await target.prepare_prefix_kv_segments(
                    target.node_id, SYSTEM_MESSAGE,
                    example.target_user_template(explicit_reasoning=args.explicit_reasoning),
                )
                # Call the official core directly.  The public wrapper is
                # tenacity-decorated and otherwise hides the useful original
                # exception inside ``RetryError`` after its single attempt.
                result = await target.agen_generate(
                    messages=message,
                    max_tokens=args.max_new_tokens,
                    request_uid=request_uid,
                    mode="kv_reuse",
                    agent_id=target.node_id,
                    agent_name="target",
                    agent_role="target",
                    # ``qwen3-1.7b`` itself has a suffix, which RelayCaching's
                    # helper treats as a file path.  Pass the latency filename
                    # explicitly rather than the result directory.
                    output_dir=output_dir / "Latency.json",
                )
                # Generation continues an unfinished assistant message.  The
                # returned text is consequently only e.g. ``B}``, while answer
                # parsing must see the complete ``\\boxed{B}`` form.
                completed_text = result.text if args.explicit_reasoning else BOXED_PREFILL + result.text
                prediction = parse_prediction(example.dataset, completed_text)
                is_correct = prediction == str(example.gold).strip().upper()
                correct += int(is_correct)
                completed += 1
                row: dict[str, Any] = {
                    **example.as_metadata(explicit_reasoning=args.explicit_reasoning),
                    "method": args.method,
                    "prediction": prediction,
                    "correct": is_correct,
                    "output": result.text,
                    "output_with_prefill": completed_text,
                    "elapsed_s": time.perf_counter() - started,
                    "relay_metadata": result.metadata,
                    **boundaries,
                }
                print(
                    f"  output={result.text!r} completed={completed_text!r} "
                    f"prediction={prediction!r} gold={example.gold!r}",
                    flush=True,
                )
            except Exception as error:
                failures += 1
                row = {
                    **example.as_metadata(explicit_reasoning=args.explicit_reasoning), "method": args.method,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "elapsed_s": time.perf_counter() - started,
                }
                print(f"  ERROR: {row['error']}", flush=True)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            LLMChat.cleanup_message_kv_cache(message)
            print(
                f"[{index}/{len(examples)}] {example.dataset} {example.source}_to_{example.target} "
                f"completed={completed} failures={failures} "
                f"accuracy={(correct / completed) if completed else float('nan'):.3f}",
                flush=True,
            )
    summary = {
        "method": args.method, "model": args.model,
        "input": str(args.input), "records": len(records), "evaluated_directions": len(examples),
        "use_debug_selection_direction": args.use_debug_selection_direction,
        "completed": completed, "failures": failures,
        "correct": correct,
        "accuracy": (correct / completed) if completed else None,
        "explicit_reasoning": args.explicit_reasoning,
        "adapter": "static shared_block prefetched under source prefix; official RelayCaching placeholder path",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> int:
    args = parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
