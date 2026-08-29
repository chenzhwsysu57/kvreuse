#!/usr/bin/env python3
"""Run the no-reuse dataset-validity baseline with official Qwen3 prompts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_MESSAGE = "You're a helpful assistant."
MODEL_IDS = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
    "8b": "Qwen/Qwen3-8B",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_user_message(record: dict[str, Any], side: str) -> str:
    prefix = record[f"prefix_{side}"]
    return f"{prefix}\n\n{record['shared_block']}\n\n{record['question']}"


def parse_prediction(dataset: str, output: str) -> str | None:
    text = output.strip()
    if dataset == "harmbench_contextual":
        match = re.match(r"^\s*(?:intent\s*[:=]\s*)?(BENIGN|HARMFUL)\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        explicit = re.findall(
            r"(?:intent|classification|answer)\s*(?:is|:|=)\s*(BENIGN|HARMFUL)\b",
            text,
            re.IGNORECASE,
        )
        return explicit[-1].upper() if explicit else None
    if dataset == "helpsteer2":
        # Endpoint-only construction: intermediate scores are invalid rather
        # than successfully parsed-but-wrong predictions.
        match = re.match(r"^\s*(?:score\s*[:=]\s*)?([04])(?!\d)", text, re.IGNORECASE)
        if match:
            return match.group(1)
        explicit = re.findall(r"(?:score|rating|answer)\s*(?:is|:|=)\s*([04])(?!\d)", text, re.IGNORECASE)
        return explicit[-1] if explicit else None

    upper = text.upper()
    # ArgKP sometimes starts with "[D] <quoted argument>"; this remains clear.
    start = re.match(r"^\s*\[?([A-H])\]?(?:\s|$|[.)])", upper)
    if start:
        return start.group(1)
    # Do not treat the agent names "Agent A/B" in utility calculations as predictions.
    explicit = re.findall(
        r"(?:FINAL(?:\s+ANSWER)?|ANSWER|OPTION|ALLOCATION|CHOICE)\s*(?:IS|:|=)\s*\[?([A-H])\]?",
        upper,
    )
    if explicit:
        return explicit[-1]
    end = re.search(r"(?:^|\n)\s*\[?([A-H])\]?[.!]?\s*$", upper)
    return end.group(1) if end else None


def split_qwen3_output(tokenizer: Any, token_ids: list[int], thinking: bool) -> tuple[str, str]:
    """Return (reasoning, final answer) using Qwen3's documented </think> split."""
    raw = tokenizer.decode(token_ids, skip_special_tokens=True)
    if not thinking:
        return "", raw.strip()
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    try:
        split_at = len(token_ids) - token_ids[::-1].index(think_end_id)
    except ValueError:
        # No closing token means generation ended inside reasoning; there is no final answer.
        return raw.strip(), ""
    reasoning = tokenizer.decode(token_ids[:split_at], skip_special_tokens=True).strip()
    answer = tokenizer.decode(token_ids[split_at:], skip_special_tokens=True).strip()
    return reasoning, answer


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_existing(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not path.exists():
        return [], set()
    rows = load_records(path)
    keys = {(row["task_id"], row["side"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError(f"duplicate task/side rows in {path}")
    return rows, keys


def write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace a JSONL file without exposing a partially written result."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def prepare_truncation_retries(
    predictions_path: Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    """Archive and remove truncated rows so only those rows are regenerated."""
    truncated = [row for row in rows if row.get("hit_max_new_tokens")]
    retry_keys = {(row["task_id"], row["side"]) for row in truncated}
    if not truncated:
        return rows, retry_keys

    archive_path = predictions_path.with_name("truncated_attempts.jsonl")
    archived_keys: set[tuple[str, str, int]] = set()
    if archive_path.exists():
        for row in load_records(archive_path):
            archived_keys.add(
                (row["task_id"], row["side"], int(row.get("generation_max_new_tokens", len(row["output_token_ids"]))))
            )
    with archive_path.open("a", encoding="utf-8") as archive:
        for row in truncated:
            archive_key = (
                row["task_id"],
                row["side"],
                int(row.get("generation_max_new_tokens", len(row["output_token_ids"]))),
            )
            if archive_key not in archived_keys:
                archive.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    retained = [row for row in rows if (row["task_id"], row["side"]) not in retry_keys]
    write_rows_atomic(predictions_path, retained)
    return retained, retry_keys


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["side"])].append(row)
        groups[(row["dataset"], "both")].append(row)
        groups[("all", "both")].append(row)
    summary = {}
    for (dataset, side), values in sorted(groups.items()):
        correct = sum(bool(value["correct"]) for value in values)
        parsed = sum(value["prediction"] is not None for value in values)
        summary[f"{dataset}/{side}"] = {
            "outputs": len(values),
            "parsed": parsed,
            "correct": correct,
            "accuracy": correct / len(values),
            "parse_rate": parsed / len(values),
            "mean_elapsed_seconds": sum(value["elapsed_seconds"] for value in values) / len(values),
            "hit_max_new_tokens": sum(bool(value["hit_max_new_tokens"]) for value in values),
        }
    return summary


def validate_model_files(model_path: Path) -> None:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        missing = sorted({name for name in index["weight_map"].values() if not (model_path / name).is_file()})
        if missing:
            raise FileNotFoundError(f"model cache is incomplete; missing {missing}")
    incomplete = list(model_path.glob("*.incomplete"))
    if incomplete:
        raise FileNotFoundError(f"model cache contains incomplete files: {incomplete}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--input", type=Path, default=Path("data/validation/second_step_10_each.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("results/full_recompute_validation"))
    parser.add_argument("--max-new-tokens", type=int, default=512, help="non-thinking output limit")
    parser.add_argument(
        "--retry-truncated",
        action="store_true",
        help="archive and regenerate only existing rows that hit max_new_tokens",
    )
    parser.add_argument(
        "--retry-max-new-tokens",
        type=int,
        default=1024,
        help="output limit used only when regenerating a truncated row",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    if args.retry_truncated and args.retry_max_new_tokens <= args.max_new_tokens:
        raise ValueError("--retry-max-new-tokens must be greater than --max-new-tokens")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation run")
    records = load_records(args.input)
    counts = defaultdict(int)
    for record in records:
        counts[record["dataset"]] += 1
    if not counts or set(counts.values()) != {10}:
        raise ValueError(f"expected 10 records from every included dataset, got {dict(counts)}")

    model_id = MODEL_IDS[args.model]
    model_path = Path(snapshot_download(model_id, local_files_only=not args.allow_download))
    validate_model_files(model_path)
    output_dir = args.output_root / f"qwen3-{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    run_config = {
        "condition": "full_recompute",
        "model_id": model_id,
        "model_path": str(model_path),
        "input_path": str(args.input),
        "input_sha256": file_sha256(args.input),
        "dataset_counts": dict(sorted(counts.items())),
        "expected_outputs": len(records) * 2,
        "system_message": SYSTEM_MESSAGE,
        "chat_template": "tokenizer.apply_chat_template",
        "enable_thinking": False,
        "reasoning_policy": "Any reasoning must be explicitly requested in the user prompt and emitted as normal output.",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "truncation_retry_max_new_tokens": args.retry_max_new_tokens,
        "truncation_retry_policy": "archive_and_regenerate_truncated_rows",
        "torch_version": torch.__version__,
    }
    if args.overwrite:
        predictions_path.unlink(missing_ok=True)
    if config_path.exists() and not args.overwrite:
        old = json.loads(config_path.read_text(encoding="utf-8"))
        # The retry fields were added after the first endpoint run. Permit that
        # metadata-only migration while still rejecting changes to core inference.
        migration_fields = {
            "truncation_retry_max_new_tokens",
            "truncation_retry_policy",
            "dataset_counts",
            "expected_outputs",
        }
        comparable = {key: old.get(key) for key in run_config if key not in migration_fields}
        expected = {key: value for key, value in run_config.items() if key not in migration_fields}
        if comparable != expected:
            raise ValueError("existing run_config differs; use --overwrite or another output root")
    config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    rows, completed = load_existing(predictions_path)
    retry_keys: set[tuple[str, str]] = set()
    if args.retry_truncated:
        rows, retry_keys = prepare_truncation_retries(predictions_path, rows)
        completed = {(row["task_id"], row["side"]) for row in rows}
        if retry_keys:
            print(
                f"Retrying {len(retry_keys)} truncated row(s) with max_new_tokens="
                f"{args.retry_max_new_tokens}",
                flush=True,
            )

    total = len(records) * 2
    if len(completed) == total:
        summary = {
            "config": run_config,
            "groups": summarize(rows),
            "completed_outputs": len(rows),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"All {total} outputs already complete for {model_id}; no model load needed.", flush=True)
        return 0

    print(f"Loading {model_id} from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device

    with predictions_path.open("a", encoding="utf-8") as output:
        done = len(completed)
        for record in records:
            for side in ("a", "b"):
                key = (record["task_id"], side)
                if key in completed:
                    continue
                user_message = build_user_message(record, side)
                messages = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_message},
                ]
                thinking = False
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
                encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                torch.cuda.synchronize()
                started = time.perf_counter()
                generation_limit = (
                    args.retry_max_new_tokens if key in retry_keys else args.max_new_tokens
                )
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": generation_limit,
                    "do_sample": False,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                sample_seed = None
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_kwargs,
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                new_ids = generated[0, input_ids.shape[1] :].tolist()
                output_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                reasoning_text, answer_text = split_qwen3_output(tokenizer, new_ids, thinking)
                prediction = parse_prediction(record["dataset"], answer_text)
                gold = record[f"gold_{side}"]
                row = {
                    "condition": "full_recompute",
                    "model_id": model_id,
                    "dataset": record["dataset"],
                    "task_id": record["task_id"],
                    "side": side,
                    "prefix": record[f"prefix_{side}"],
                    "shared_block": record["shared_block"],
                    "question": record["question"],
                    "gold": gold,
                    "messages": messages,
                    "rendered_input": rendered,
                    "input_token_ids": input_ids[0].tolist(),
                    "input_tokens": input_ids.shape[1],
                    "output_text": output_text,
                    "reasoning_text": reasoning_text,
                    "answer_text": answer_text,
                    "output_token_ids": new_ids,
                    "output_tokens": len(new_ids),
                    "generation_max_new_tokens": generation_limit,
                    "enable_thinking": thinking,
                    "sample_seed": sample_seed,
                    "hit_max_new_tokens": len(new_ids) == generation_kwargs["max_new_tokens"],
                    "prediction": prediction,
                    "correct": prediction == gold,
                    "elapsed_seconds": elapsed,
                }
                output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                output.flush()
                rows.append(row)
                completed.add(key)
                done += 1
                print(
                    f"[{done:02d}/{total}] {record['dataset']} {record['task_id']} side={side} "
                    f"gold={gold} pred={prediction!r} answer={answer_text!r}",
                    flush=True,
                )

    summary = {
        "config": run_config,
        "groups": summarize(rows),
        "completed_outputs": len(rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["groups"], indent=2), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
