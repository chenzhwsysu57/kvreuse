#!/usr/bin/env python3
"""Evaluate boundary-difference KV calibration without rerunning baselines.

For a source block, remove its absolute prefix-dependent offset and re-anchor
the block to the target prefix's last token. Keys are de-RoPE'd before this
operation and rotated at the target positions afterwards; Values use the same
first-token anchored residual in their native space.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Any

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import run_direct_reuse as dr
except ModuleNotFoundError:  # also support ``import scripts...`` from repo root
    from . import run_direct_reuse as dr


def calibrate_block(model: Any, source: list[tuple[torch.Tensor, torch.Tensor]],
                    target_prefix: list[tuple[torch.Tensor, torch.Tensor]],
                    source_start: int, target_start: int,
                    source_prefix: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
                    method: str = "first_anchor") -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Calibrate a block by preserving residuals and changing its anchor.

    ``first_anchor`` uses the first block token. ``prefix_offset`` is the
    KVCOMM-style variant: estimate the prefix-dependent offset from the last
    token of the source and target prefixes, then apply that offset to every
    source block token.
    """
    length = source[0][0].shape[-2]
    src_pos = torch.arange(source_start, source_start + length, device=source[0][0].device)
    tgt_pos = torch.arange(target_start, target_start + length, device=source[0][0].device)
    source_anchor_pos = torch.tensor([source_start - 1], device=source[0][0].device)
    anchor_pos = torch.tensor([target_start - 1], device=source[0][0].device)
    out = []
    for (sk, sv), (pk, pv) in zip(source, target_prefix):
        # rotary_emb returns [1, 1, seq, dim] cos/sin, matching the cache.
        cos, sin = dr.rope_cos_sin(model, sk, torch.cat((src_pos, source_anchor_pos, anchor_pos, tgt_pos)))
        cs = cos[..., :length, :]; csa = cos[..., length:length + 1, :]
        ca = cos[..., length + 1:length + 2, :]; ct = cos[..., length + 2:, :]
        ss = sin[..., :length, :]; ssa = sin[..., length:length + 1, :]
        sa = sin[..., length + 1:length + 2, :]; st = sin[..., length + 2:, :]
        skf = sk.float(); pkf = pk[..., -1:, :].float()
        src_unrot = skf * cs - dr.rotate_half_qwen(skf) * ss
        anchor_unrot = pkf * ca - dr.rotate_half_qwen(pkf) * sa
        if method == "prefix_offset":
            if source_prefix is None:
                raise ValueError("prefix_offset requires source_prefix")
            spk = source_prefix[len(out)][0][..., -1:, :].float()
            src_anchor = spk * csa - dr.rotate_half_qwen(spk) * ssa
            calibrated_unrot = anchor_unrot + (src_unrot - src_anchor)
        else:
            calibrated_unrot = anchor_unrot + (src_unrot - src_unrot[..., :1, :])
        calibrated_key = calibrated_unrot * ct + dr.rotate_half_qwen(calibrated_unrot) * st
        svf = sv.float(); pvf = pv[..., -1:, :].float()
        if method == "prefix_offset":
            spv = source_prefix[len(out)][1][..., -1:, :].float()
            calibrated_value = pvf + (svf - spv)
        else:
            calibrated_value = pvf + (svf - svf[..., :1, :])
        out.append((calibrated_key.to(sk.dtype).contiguous(), calibrated_value.to(sv.dtype).contiguous()))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=dr.MODEL_IDS, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True,
                   help="existing full/reuse samples.jsonl; it is read only")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--explicit-reasoning", action="store_true",
                   help="match the direct runner's visible-reasoning prompt and boxed-answer parsing")
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--method", choices=("first_anchor", "prefix_offset"), default="first_anchor",
                   help="prefix_offset is the KVCOMM-style prefix-anchor calibration")
    args = p.parse_args()
    records = dr.load_jsonl(args.input)
    base_rows = {r["task_id"]: r for r in dr.load_jsonl(args.baseline)}
    records = [r for r in records if r["task_id"] in base_rows]
    if args.max_samples > 0: records = records[:args.max_samples]
    if not records: raise ValueError("no input records have matching baseline rows")
    outdir = args.output_root / f"qwen3-{args.model}"
    outdir.mkdir(parents=True, exist_ok=True)
    pred_path, summary_path = outdir / "samples.jsonl", outdir / "summary.json"
    if args.overwrite: pred_path.unlink(missing_ok=True); summary_path.unlink(missing_ok=True)
    done = {r["task_id"] for r in dr.load_jsonl(pred_path)} if pred_path.exists() else set()
    model_path = Path(snapshot_download(dr.MODEL_IDS[args.model], local_files_only=not args.allow_download))
    dr.validate_model_files(model_path)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16,
                                                  device_map="cuda:0", attn_implementation="sdpa").eval()
    dr.assert_default_rope(model)
    for idx, rec in enumerate(records, 1):
        if rec["task_id"] in done: continue
        parts = {s: dr.build_prompt_parts(tok, rec, s, enable_thinking=False,
                    explicit_reasoning=args.explicit_reasoning, boxed_output=True) for s in ("a", "b")}
        if not torch.equal(parts["a"].block_ids, parts["b"].block_ids):
            raise ValueError(f"{rec['task_id']}: block tokenization differs")
        full, prefixes = {}, {}
        for side in ("a", "b"):
            _, full[side] = dr.forward_ids(model, parts[side].full_ids)
            _, prefixes[side] = dr.forward_ids(model, parts[side].prefix_ids)
        calibrated = {}
        for name, source, target in (("a_to_b", "a", "b"), ("b_to_a", "b", "a")):
            block = dr.slice_cache(full[source], parts[source].block_token_start, parts[source].block_token_end)
            block = calibrate_block(model, block, prefixes[target], parts[source].block_token_start,
                                    parts[target].block_token_start, prefixes[source], args.method)
            mixed = dr.splice_prefix_block(prefixes[target], block)
            logits, layers = dr.forward_suffix(model, mixed, parts[target].suffix_ids)
            started = time.perf_counter()
            ids, hit = dr.greedy_continue(model, tok, logits, layers, args.max_new_tokens)
            elapsed = time.perf_counter() - started
            calibrated[name] = dr.result_from_generation(
                tok, rec["dataset"], rec[f"gold_{target}"], ids, hit,
                elapsed, args.max_new_tokens, False, args.explicit_reasoning,
                "" if args.explicit_reasoning else "The answer is: \\boxed{",
            )
        row = {"task_id": rec["task_id"], "dataset": rec["dataset"], "calibration": calibrated,
               "baseline": {"full": base_rows[rec["task_id"]].get("full", {}),
                            "reuse": base_rows[rec["task_id"]].get("reuse", {})}}
        dr.append_jsonl(pred_path, row); print(f"[{idx}/{len(records)}] {rec['task_id']}", flush=True)
        del full, prefixes, calibrated; dr.sync_cuda()
    rows = dr.load_jsonl(pred_path)
    cal_values = [v for r in rows for v in r["calibration"].values()]
    full_values = [r["baseline"].get("full", {}).get(side, {}) for r in rows for side in ("a", "b")]
    reuse_values = [r["baseline"].get("reuse", {}).get(direction, {}) for r in rows for direction in ("a_to_b", "b_to_a")]
    cal_acc = sum(v.get("correct", False) for v in cal_values) / max(1, len(cal_values))
    full_acc = sum(v.get("correct", False) for v in full_values) / max(1, len(full_values))
    reuse_acc = sum(v.get("correct", False) for v in reuse_values) / max(1, len(reuse_values))
    metrics = {"outputs": len(cal_values), "accuracy": cal_acc,
               "baseline_full_accuracy": full_acc,
               "baseline_cross_reuse_accuracy": reuse_acc,
               "calibration_minus_full_accuracy": cal_acc - full_acc,
               "calibration_minus_cross_reuse_accuracy": cal_acc - reuse_acc,
               "by_direction": {d: {"accuracy": sum(r["calibration"][d]["correct"] for r in rows) / max(1, len(rows)),
                                    "parse_rate": sum(r["calibration"][d]["prediction"] is not None for r in rows) / max(1, len(rows))}
                                for d in ("a_to_b", "b_to_a")}}
    dr.write_json_atomic(summary_path, {"method": args.method, "metrics": metrics,
        "baseline_path": str(args.baseline), "input_path": str(args.input), "model": dr.MODEL_IDS[args.model]})
    print(json.dumps(metrics, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
