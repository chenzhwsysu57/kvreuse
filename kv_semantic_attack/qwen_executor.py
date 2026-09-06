"""Qwen3 adapter built on the repository's direct-reuse implementation."""

from pathlib import Path
import re
from datetime import datetime
from typing import Any

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.run_direct_reuse import (
    MODEL_IDS,
    build_prompt_parts,
    extract_cache_tensors,
    forward_ids,
    forward_suffix,
    greedy_continue,
    make_dynamic_cache,
    input_device,
    relocate_block,
    slice_cache,
    splice_prefix_block,
    assert_default_rope,
)

from .executor import Executor
from .schemas import AttackCase, ExecutionResult
from .step_logger import StepLogger


class Qwen3KVReuseExecutor(Executor):
    """Run synthetic two-context cases through the real Qwen3 cache path.

    This adapter intentionally uses the same prompt builder and RoPE relocation
    functions as ``scripts/run_direct_reuse.py``. It does not write benchmark
    artifacts; self-play round JSON is written by the orchestrator.
    """

    def __init__(self, *, model_size: str = "0.6b", reasoning: bool = False,
                 trace_dir: str | None = None, allow_download: bool = False,
                 debug: bool = False, step_logger: StepLogger | None = None):
        if model_size not in MODEL_IDS:
            raise ValueError(f"unsupported Qwen3 model size: {model_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3KVReuseExecutor requires CUDA")
        print(f"[qwen-executor] resolving {MODEL_IDS[model_size]}", flush=True)
        model_path = Path(snapshot_download(
            MODEL_IDS[model_size], local_files_only=not allow_download
        ))
        self.model_path = model_path
        self.reasoning = reasoning
        self.max_new_tokens = 1024 if reasoning else 32
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.step_logger = step_logger
        self.debug = debug
        self._trace_index = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        self.model.eval()
        assert_default_rope(self.model)
        print(f"[qwen-executor] loaded model from {model_path}", flush=True)

    @staticmethod
    def _record(instruction: str, shared_block: str, query: str) -> dict[str, str]:
        return {
            "prefix_a": instruction,
            "prefix_b": instruction,
            "shared_block": shared_block,
            "question": query,
        }

    def _parts(self, instruction: str, block: str, query: str):
        # Answer-format instructions belong to the execution prompt, not to
        # the attacker's case. The attacker supplies only the plain gold
        # label (for example, "A"); the model is instructed here to emit its
        # answer as exactly one boxed answer.
        return build_prompt_parts(
            self.tokenizer,
            self._record(instruction, block, query),
            "a",
            enable_thinking=False,
            explicit_reasoning=self.reasoning,
            boxed_output=True,
        )

    def _save_execution_trace(self, kind: str, inputs: dict[str, str], output: str) -> None:
        if self.trace_dir is None and self.step_logger is None:
            return
        input_text = "\n\n".join(
            f"===== {name} =====\n{text}" for name, text in inputs.items()
        )
        content = (
            f"timestamp: {datetime.now().astimezone().isoformat()}\n"
            f"mode: {'reasoning' if self.reasoning else 'no-reasoning'}\n"
            f"max_new_tokens: {self.max_new_tokens}\n\n"
            f"{input_text}\n\n===== MODEL OUTPUT =====\n{output}\n"
        )
        if self.step_logger is not None:
            self.step_logger.write(f"executor_{kind}", content)
        else:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self._trace_index += 1
            safe_kind = re.sub(r"[^A-Za-z0-9_.-]+", "_", kind)
            path = self.trace_dir / f"{self._trace_index:04d}_{safe_kind}.txt"
            path.write_text(content, encoding="utf-8")

    def _debug_execution_input(self, kind: str, inputs: dict[str, str]) -> None:
        if not self.debug:
            return
        print(f"\n===== EXECUTOR INPUT ({kind}) =====", flush=True)
        for name, text in inputs.items():
            print(f"===== {name} =====\n{text}\n", flush=True)

    def _debug_execution_output(self, kind: str, result: ExecutionResult) -> None:
        if self.debug:
            print(f"===== EXECUTOR OUTPUT ({kind}) =====", flush=True)
            print(result.text, flush=True)
            print(f"metadata={result.metadata}", flush=True)
            print("===== END EXECUTOR CALL =====\n", flush=True)

    @torch.inference_mode()
    def _greedy_continue_batch(self, initial_logits: torch.Tensor,
                               past_layers: list[tuple[torch.Tensor, torch.Tensor]],
                               lengths: list[int], max_new_tokens: int,
                               base_attention: torch.Tensor | None = None):
        """Greedy-decode all full requests together using one batched cache."""
        device = input_device(self.model)
        batch = initial_logits.shape[0]
        cache = make_dynamic_cache(self.model, past_layers)
        padded_length = past_layers[0][0].shape[-2]
        if base_attention is None:
            base_attention = torch.ones(
                (batch, padded_length), device=device, dtype=torch.long
            )
        eos = self.tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        current = initial_logits.argmax(dim=-1).view(batch, 1).to(device)
        generated = [[int(current[row, 0].item())] for row in range(batch)]
        done = torch.tensor(
            [generated[row][0] in eos_ids for row in range(batch)],
            device=device, dtype=torch.bool,
        )
        for step in range(1, max_new_tokens):
            if bool(done.all()):
                break
            position_ids = torch.tensor(
                [[lengths[row] + step - 1] for row in range(batch)],
                device=device, dtype=torch.long,
            )
            # Left padding is masked out; generated tokens occupy the common
            # right edge of the batched cache.
            attention = torch.ones(
                (batch, padded_length + step), device=device, dtype=torch.long
            )
            attention[:, :padded_length] = base_attention
            output = self.model(
                input_ids=current,
                attention_mask=attention,
                position_ids=position_ids,
                cache_position=torch.tensor([padded_length + step - 1], device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            next_token = output.logits[:, -1].argmax(dim=-1)
            next_token = torch.where(done, torch.tensor(eos or 0, device=device), next_token)
            for row in range(batch):
                generated[row].append(int(next_token[row].item()))
            done |= torch.tensor(
                [int(next_token[row].item()) in eos_ids for row in range(batch)],
                device=device, dtype=torch.bool,
            )
            current = next_token.view(batch, 1)
        return generated

    @torch.inference_mode()
    def _prefill_ids_batch(self, ids_list: list[torch.Tensor], pad_to_length: int | None = None):
        """Right-align variable-length sequences and return trimmed metadata."""
        device = input_device(self.model)
        lengths = [int(ids.numel()) for ids in ids_list]
        width = max(max(lengths), pad_to_length or 0)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer needs pad_token_id or eos_token_id for batching")
        input_ids = torch.full((len(ids_list), width), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros((len(ids_list), width), dtype=torch.long, device=device)
        positions = torch.zeros((len(ids_list), width), dtype=torch.long, device=device)
        for row, ids in enumerate(ids_list):
            n = lengths[row]
            input_ids[row, width - n:] = ids.to(device)
            attention[row, width - n:] = 1
            positions[row, width - n:] = torch.arange(n, device=device)
        output = self.model(
            input_ids=input_ids, attention_mask=attention,
            position_ids=positions, cache_position=torch.arange(width, device=device),
            use_cache=True, return_dict=True,
        )
        layers = extract_cache_tensors(output.past_key_values)
        logits = output.logits[torch.arange(len(ids_list), device=device),
                               torch.tensor([width - 1] * len(ids_list), device=device)].detach().float()
        trimmed = [
            [(key[row:row + 1, ..., width - lengths[row]:, :].contiguous(),
              value[row:row + 1, ..., width - lengths[row]:, :].contiguous())
             for key, value in layers]
            for row in range(len(ids_list))
        ]
        return logits, trimmed

    @torch.inference_mode()
    def _run_reuse_batch(self, requests: list[dict[str, str]],
                         pad_widths: tuple[int, int, int, int] | None = None) -> list[ExecutionResult]:
        """Batch reuse requests while preserving per-row KV relocation metadata."""
        if not requests:
            return []
        source_parts = [self._parts(x["donor_prefix"], x["shared_block"], x["question"])
                        for x in requests]
        target_query = [f'{x["remedy"].strip()}\n\n{x["question"]}'
                        if x["remedy"].strip() else x["question"] for x in requests]
        target_parts = [self._parts(x["target_prefix"], x["shared_block"], query)
                        for x, query in zip(requests, target_query)]
        for source, target in zip(source_parts, target_parts):
            if not torch.equal(source.block_ids, target.block_ids):
                raise ValueError("source/target shared block is not token-aligned identically")

        source_pad = pad_widths[0] if pad_widths else None
        target_pad = pad_widths[1] if pad_widths else None
        suffix_pad = pad_widths[2] if pad_widths else None
        past_pad = pad_widths[3] if pad_widths else None
        _, source_layers = self._prefill_ids_batch([x.full_ids for x in source_parts], source_pad)
        _, target_layers = self._prefill_ids_batch([x.prefix_ids for x in target_parts], target_pad)
        mixed_rows = []
        logical_lengths = []
        for row, (source, target) in enumerate(zip(source_parts, target_parts)):
            source_block = slice_cache(source_layers[row], source.block_token_start,
                                       source.block_token_end)
            relocated = relocate_block(self.model, source_block,
                                       source.block_token_start,
                                       target.block_token_start)
            mixed = splice_prefix_block(target_layers[row], relocated)
            mixed_rows.append(mixed)
            logical_lengths.append(mixed[0][0].shape[-2])

        max_past = max(max(logical_lengths), past_pad or 0)
        padded_layers = []
        for layer_idx in range(len(mixed_rows[0])):
            keys, values = [], []
            for row, mixed in enumerate(mixed_rows):
                pad = max_past - logical_lengths[row]
                key, value = mixed[layer_idx]
                keys.append(torch.nn.functional.pad(key, (0, 0, pad, 0)))
                values.append(torch.nn.functional.pad(value, (0, 0, pad, 0)))
            padded_layers.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        suffix_ids = [target.suffix_ids for target in target_parts]
        device = input_device(self.model)
        suffix_width = max(max(int(ids.numel()) for ids in suffix_ids), suffix_pad or 0)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        suffix = torch.full((len(requests), suffix_width), pad_id,
                            dtype=torch.long, device=device)
        suffix_attention = torch.zeros((len(requests), suffix_width),
                                       dtype=torch.long, device=device)
        suffix_positions = torch.zeros((len(requests), suffix_width),
                                       dtype=torch.long, device=device)
        for row, ids in enumerate(suffix_ids):
            n = int(ids.numel())
            suffix[row, :n] = ids.to(device)
            suffix_attention[row, :n] = 1
            suffix_positions[row, :n] = torch.arange(
                logical_lengths[row], logical_lengths[row] + n, device=device
            )
        attention = torch.zeros((len(requests), max_past + suffix_width),
                                dtype=torch.long, device=device)
        for row, length in enumerate(logical_lengths):
            attention[row, max_past - length:max_past] = 1
            attention[row, max_past:] = suffix_attention[row]
        output = self.model(
            input_ids=suffix, attention_mask=attention,
            position_ids=suffix_positions,
            cache_position=torch.arange(max_past, max_past + suffix_width, device=device),
            past_key_values=make_dynamic_cache(self.model, padded_layers),
            use_cache=True, return_dict=True,
        )
        last = torch.tensor([n - 1 for n in map(lambda x: int(x.numel()), suffix_ids)], device=device)
        logits = output.logits[torch.arange(len(requests), device=device), last].detach().float()
        suffix_layers = extract_cache_tensors(output.past_key_values)
        generated = self._greedy_continue_batch(
            logits, suffix_layers,
            [logical_lengths[row] + int(suffix_ids[row].numel()) for row in range(len(requests))],
            self.max_new_tokens,
            base_attention=attention,
        )
        results = []
        for row, request in enumerate(requests):
            result = self._decode_result(
                generated[row], request["gold"],
                prefix="" if self.reasoning else r"The answer is: \boxed{",
            )
            result.metadata.update({"batched_prefill": True,
                                    "batched_generation": True,
                                    "batched_reuse": True,
                                    "remedy": request["remedy"],
                                    "direction": request["label"]})
            self._save_execution_trace(request["label"], {
                "donor_source_prompt": source_parts[row].rendered,
                "target_reuse_prompt": target_parts[row].rendered,
            }, result.text)
            results.append(result)
        return results

    @torch.inference_mode()
    def run_mixed_batch(self, requests: list[dict[str, str]],
                        pad_to_length: int | None = None) -> list[ExecutionResult]:
        """Decode full and already-assembled reuse contexts in one batch.

        Each row is first reduced to its own final prompt KV.  Full rows use a
        normal prefill; reuse rows use donor-block extraction, relocation, and
        splicing.  Only after that heterogeneous preparation do we pad and
        stack the KV rows for one common batched decoding loop.
        """
        if not requests:
            return []
        prepared = []
        for request in requests:
            mode = request["mode"]
            if mode == "full":
                parts = self._parts(request["prefix"], request["shared_block"],
                                    request["question"])
                logits, layers = forward_ids(self.model, parts.full_ids)
                prepared.append((logits, layers, int(parts.full_ids.numel()),
                                 parts.rendered))
                continue
            if mode != "reuse":
                raise ValueError(f"unknown mixed batch mode: {mode!r}")
            source = self._parts(request["donor_prefix"], request["shared_block"],
                                 request["question"])
            query = (f'{request["remedy"].strip()}\n\n{request["question"]}'
                     if request["remedy"].strip() else request["question"])
            target = self._parts(request["target_prefix"], request["shared_block"], query)
            if not torch.equal(source.block_ids, target.block_ids):
                raise ValueError("source/target shared block is not token-aligned identically")
            _, source_layers = forward_ids(self.model, source.full_ids)
            _, target_prefix_layers = forward_ids(self.model, target.prefix_ids)
            source_block = slice_cache(source_layers, source.block_token_start,
                                       source.block_token_end)
            relocated = relocate_block(self.model, source_block,
                                       source.block_token_start, target.block_token_start)
            mixed = splice_prefix_block(target_prefix_layers, relocated)
            logits, final_layers = forward_suffix(self.model, mixed, target.suffix_ids)
            prepared.append((logits, final_layers,
                             int(mixed[0][0].shape[-2] + target.suffix_ids.numel()),
                             target.rendered))

        device = input_device(self.model)
        lengths = [item[2] for item in prepared]
        max_length = max(max(lengths), pad_to_length or 0)
        padded_layers = []
        for layer_idx in range(len(prepared[0][1])):
            keys, values = [], []
            for _, layers, length, _ in prepared:
                key, value = layers[layer_idx]
                pad = max_length - length
                keys.append(torch.nn.functional.pad(key, (0, 0, pad, 0)))
                values.append(torch.nn.functional.pad(value, (0, 0, pad, 0)))
            padded_layers.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        attention = torch.zeros((len(prepared), max_length),
                                dtype=torch.long, device=device)
        for row, length in enumerate(lengths):
            attention[row, max_length - length:] = 1
        logits = torch.cat([item[0].view(1, -1) for item in prepared], dim=0)
        generated = self._greedy_continue_batch(
            logits, padded_layers, lengths, self.max_new_tokens,
            base_attention=attention,
        )
        results = []
        for row, (request, item) in enumerate(zip(requests, prepared)):
            result = self._decode_result(
                generated[row], request["gold"],
                prefix="" if self.reasoning else r"The answer is: \boxed{",
            )
            result.metadata.update({"mixed_batch": True,
                                    "batch_size": len(requests),
                                    "batched_generation": True,
                                    "label": request["label"]})
            self._save_execution_trace(request["label"],
                {"assembled_prompt": item[3]}, result.text)
            results.append(result)
        return results

    @torch.inference_mode()
    def run_full_batch(self, requests: list[dict[str, str]],
                       pad_to_length: int | None = None) -> list[ExecutionResult]:
        """Run independent full requests with native Transformers generation."""
        if not requests:
            return []
        print(f"[qwen-executor] full batch generate: batch_size={len(requests)}", flush=True)
        parts = [self._parts(x["prefix"], x["shared_block"], x["question"])
                 for x in requests]
        device = input_device(self.model)
        lengths = [int(x.full_ids.numel()) for x in parts]
        width = max(max(lengths), pad_to_length or 0)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer needs pad_token_id or eos_token_id for batching")
        input_ids = torch.full((len(parts), width), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros((len(parts), width), dtype=torch.long, device=device)
        # Left padding is the convention supported by decoder-only
        # model.generate for variable-length prompts.  Crucially, we do not
        # provide custom cache_position/position_ids here: Qwen computes them.
        for row, item in enumerate(parts):
            n = lengths[row]
            input_ids[row, width - n:] = item.full_ids.to(device)
            attention[row, width - n:] = 1
        sequences = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        results = []
        for row, request in enumerate(requests):
            tokens = sequences[row, width:].tolist()
            eos = self.tokenizer.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            hit_limit = len(tokens) >= self.max_new_tokens and (
                not eos_ids or tokens[-1] not in eos_ids
            )
            result = self._decode_result(
                tokens, request["gold"],
                prefix="" if self.reasoning else r"The answer is: \boxed{",
            )
            result.metadata["hit_max_new_tokens"] = hit_limit
            result.metadata["label"] = request["label"]
            result.metadata["batched_prefill"] = True
            result.metadata["batched_generation"] = True
            self._save_execution_trace(request["label"],
                {"rendered_prompt": parts[row].rendered}, result.text)
            results.append(result)
        del sequences, input_ids, attention
        return results

    def execute_attack(self, case: AttackCase, remedy: str) -> dict[str, ExecutionResult]:
        # Full A/B have no cache dependency and share one prefill batch. Reuse
        # paths stay explicit so donor and target KV ownership remains unambiguous.
        requests = [
            {"prefix": case.prefix_a, "shared_block": case.shared_block,
             "question": case.question, "gold": case.gold_a, "label": "full_a"},
            {"prefix": case.prefix_b, "shared_block": case.shared_block,
             "question": case.question, "gold": case.gold_b, "label": "full_b"},
        ]
        try:
            full = self.run_full_batch(requests)
        except Exception as exc:
            # Keep self-play usable if a model/version rejects the padded batch
            # layout.  The serial path is slower but has the original semantics.
            print(f"[qwen-executor] full batch failed; falling back to serial: {exc!r}", flush=True)
            full = [self.run_full(**request) for request in requests]
        reuse = self._run_reuse_batch([
            {"donor_prefix": case.prefix_b, "target_prefix": case.prefix_a,
             "shared_block": case.shared_block, "question": case.question,
             "remedy": remedy, "gold": case.gold_a, "label": "reuse_b_to_a"},
            {"donor_prefix": case.prefix_a, "target_prefix": case.prefix_b,
             "shared_block": case.shared_block, "question": case.question,
             "remedy": remedy, "gold": case.gold_b, "label": "reuse_a_to_b"},
        ])
        return {"full_a": full[0], "full_b": full[1], **reuse}

    def execute_attack_batch(self, cases: list[AttackCase], remedy: str) -> list[dict[str, ExecutionResult]]:
        """Batch all independent full prefills, then run cache-sensitive reuse.

        Full A/B requests across every candidate can share one padded prefill
        batch.  Reuse is kept per case because each row has its own donor block,
        target prefix length, and relocated cache; blindly batching those rows
        would risk mixing cache ownership.
        """
        if not cases:
            return []
        requests = []
        for index, case in enumerate(cases):
            suffix = f"_{index}"
            requests.extend([
                {"prefix": case.prefix_a, "shared_block": case.shared_block,
                 "question": case.question, "gold": case.gold_a,
                 "label": f"full_a{suffix}"},
                {"prefix": case.prefix_b, "shared_block": case.shared_block,
                 "question": case.question, "gold": case.gold_b,
                 "label": f"full_b{suffix}"},
            ])
        try:
            full = self.run_full_batch(requests)
        except Exception as exc:
            print(f"[qwen-executor] cross-case full batch failed; falling back to per-case execution: {exc!r}", flush=True)
            return [self.execute_attack(case, remedy) for case in cases]
        reuse_requests = []
        for case in cases:
            reuse_requests.extend([
                {"donor_prefix": case.prefix_b, "target_prefix": case.prefix_a,
                 "shared_block": case.shared_block, "question": case.question,
                 "remedy": remedy, "gold": case.gold_a, "label": "reuse_b_to_a"},
                {"donor_prefix": case.prefix_a, "target_prefix": case.prefix_b,
                 "shared_block": case.shared_block, "question": case.question,
                 "remedy": remedy, "gold": case.gold_b, "label": "reuse_a_to_b"},
            ])
        reuse_all = self._run_reuse_batch(reuse_requests)
        results = []
        for index, case in enumerate(cases):
            results.append({"full_a": full[2 * index], "full_b": full[2 * index + 1],
                            "reuse_b_to_a": reuse_all[2 * index],
                            "reuse_a_to_b": reuse_all[2 * index + 1]})
        return results
    def _decode_result(self, token_ids: list[int], expected: str,
                       *, prefix: str = "") -> ExecutionResult:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        text = f"{prefix}{text}" if prefix else text
        metadata = {}
        boxed = re.search(r"\\boxed\{", text)
        metadata["reasoning_text"] = text[:boxed.start()].strip() if boxed else text
        metadata.update({"model_path": str(self.model_path),
                         "reasoning_mode": self.reasoning})
        return ExecutionResult(text=text, metadata=metadata)

    @torch.inference_mode()
    def run_full(self, *, prefix: str, shared_block: str,
                 question: str, gold: str, label: str) -> ExecutionResult:
        print(f"[qwen-executor] {label}", flush=True)
        parts = self._parts(prefix, shared_block, question)
        self._debug_execution_input(label, {
            "rendered_prompt": parts.rendered,
        })
        logits, layers = forward_ids(self.model, parts.full_ids)
        tokens, hit_limit = greedy_continue(
            self.model, self.tokenizer, logits, layers, self.max_new_tokens
        )
        result = self._decode_result(
            tokens, gold,
            prefix="" if self.reasoning else r"The answer is: \boxed{",
        )
        result.metadata["hit_max_new_tokens"] = hit_limit
        result.metadata["label"] = label
        self._save_execution_trace(label, {
            "gold": gold, "rendered_prompt": parts.rendered,
        }, result.text)
        self._debug_execution_output(label, result)
        if self.debug:
            display = "prefixa" if label == "full_a" else "prefixb"
            print(f"full {display} | block ans:[{result.text}] correct: [{gold}]", flush=True)
        return result

    @torch.inference_mode()
    def run_reuse(self, *, donor_prefix: str, target_prefix: str,
                  shared_block: str, question: str, remedy: str,
                  gold: str, label: str) -> ExecutionResult:
        direction = label
        print(f"[qwen-executor] {direction}; remedy_chars={len(remedy)}", flush=True)
        source = self._parts(donor_prefix, shared_block, question)
        target_query = f"{remedy.strip()}\n\n{question}" if remedy.strip() else question
        target = self._parts(target_prefix, shared_block, target_query)
        self._debug_execution_input(
            "reuse",
            {"donor_source_prompt": source.rendered, "target_reuse_prompt": target.rendered},
        )
        if not torch.equal(source.block_ids, target.block_ids):
            raise ValueError("source/target shared block is not token-aligned identically")

        _, source_layers = forward_ids(self.model, source.full_ids)
        _, target_prefix_layers = forward_ids(self.model, target.prefix_ids)
        source_block = slice_cache(source_layers, source.block_token_start, source.block_token_end)
        relocated = relocate_block(
            self.model, source_block,
            source.block_token_start, target.block_token_start,
        )
        mixed = splice_prefix_block(target_prefix_layers, relocated)
        logits, prompt_layers = forward_suffix(self.model, mixed, target.suffix_ids)
        tokens, hit_limit = greedy_continue(
            self.model, self.tokenizer, logits, prompt_layers, self.max_new_tokens
        )
        result = self._decode_result(
            tokens, gold,
            prefix="" if self.reasoning else r"The answer is: \boxed{",
        )
        result.metadata.update({
            "hit_max_new_tokens": hit_limit,
            "source_block_tokens": int(source.block_ids.numel()),
            "target_prefix_tokens": int(target.prefix_ids.numel()),
            "remedy": remedy,
        })
        self._save_execution_trace(
            direction,
             {"gold": gold, "remedy": remedy,
             "donor_source_prompt": source.rendered, "target_reuse_prompt": target.rendered},
            result.text,
        )
        result.metadata["direction"] = direction
        self._debug_execution_output(direction, result)
        if self.debug:
            if direction == "reuse_b_to_a":
                print(f"reuse prefixa | (block|prefix b): [{result.text}] correct [{gold}]", flush=True)
            elif direction == "reuse_a_to_b":
                print(
                    f"reuse prefixb | (block|prefix a): [{result.text}] correct [{gold}]",
                    flush=True,
                )
        return result
