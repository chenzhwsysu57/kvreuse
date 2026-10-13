#!/usr/bin/env python3
"""Aggregate metrics from a single experiment result JSON (``*_{run_tag}.json``).

Reads per-sample ``Agent traces`` metadata produced by ``RelayCaching/llm/gpt_chat.py``.
Does not parse logs (no memory-overhead fields).

Usage:
    python experiments/analysis/analyze_result.py path/to/mmlu_....json
    python experiments/analysis/analyze_result.py path/to/result.json --json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root: RelayCaching_public/
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from RelayCaching.utils.metrics import compute_reuse_rate_cost, compute_ttft_for_analysis


def _mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def _std(xs: List[float]) -> float:
    return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


def _median(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def _infer_method_label(
    traces: List[Dict[str, Any]],
    cli_method: Optional[str],
) -> str:
    if cli_method:
        return cli_method.strip().lower()
    for trace in traces:
        meta = trace.get("metadata") or {}
        if meta.get("baseline_label"):
            return str(meta["baseline_label"]).lower()
        if meta.get("algorithm_execution_mode"):
            return str(meta["algorithm_execution_mode"]).lower()
    return "unknown"


def _ttft_from_metadata(meta: Dict[str, Any], method_label: str) -> Optional[float]:
    if meta.get("ttft_for_analysis") is not None:
        return float(meta["ttft_for_analysis"])
    if meta.get("generation_ttft") is None:
        return None
    exec_mode = str(
        meta.get("algorithm_execution_mode") or method_label
    ).lower()
    breakdown = meta.get("kvcomm_breakdown_timing")
    return compute_ttft_for_analysis(
        execution_mode=exec_mode,
        generation_ttft=float(meta["generation_ttft"]),
        preprocess_latency=float(meta.get("preprocess_latency") or 0.0),
        kvcomm_breakdown_timing=breakdown if isinstance(breakdown, dict) else None,
    )


def _default_layers(json_path: Path) -> Tuple[int, int, int]:
    """Return (num_layers, start_layer, end_layer) from path heuristics."""
    path_s = str(json_path)
    num_layers = 32 if ("Llama-3.1-8B" in path_s or "Llama-3" in path_s) else 28
    rec = re.search(r"rec-(\d+)-(\d+)", path_s)
    if rec:
        return num_layers, int(rec.group(1)), int(rec.group(2))
    if num_layers == 32:
        return num_layers, 1, 18
    return num_layers, 3, 22


def _sample_reuse_rate(
    *,
    method_label: str,
    algorithm_mode: str,
    acc_ph: int,
    acc_recompute: int,
    acc_epic: int,
    kvcomm_reuse: int,
    num_layers: int,
    start_layer: int,
    end_layer: int,
) -> float:
    label = method_label.lower()
    algo = (algorithm_mode or label).lower()

    if label == "zero":
        return 1.0
    if label == "full":
        return 0.0

    if algo == "kvcomm" or label == "kvcomm":
        return (kvcomm_reuse / acc_ph) if acc_ph > 0 else 0.0

    if algo == "epic" or label.startswith("epic"):
        if acc_ph <= 0:
            return 0.0
        return max(0.0, min(1.0, (acc_ph - acc_epic) / acc_ph))

    if acc_ph <= 0:
        return 0.0

    exec_mode = algo if algo in ("relaycaching", "cacheblend", "dense_prefill_only") else label
    if exec_mode not in ("relaycaching", "cacheblend", "dense_prefill_only"):
        return 0.0

    return compute_reuse_rate_cost(
        execution_mode=exec_mode,
        ph_tokens=acc_ph,
        recompute_tokens=acc_recompute,
        num_layers=num_layers,
        start_layer=start_layer,
        end_layer=end_layer,
    )


def analyze_result_json(
    json_path: Path,
    *,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON list of samples.")

    num_layers, start_layer, end_layer = _default_layers(json_path)
    ttft_list: List[float] = []
    acc_ph = 0
    acc_recompute = 0
    acc_epic = 0
    kvcomm_reuse = 0
    total_generation_tokens = 0
    method_label = "unknown"
    algorithm_mode = "unknown"
    n_samples = 0

    for sample in data:
        if not isinstance(sample, dict):
            continue
        traces = sample.get("Agent traces") or []
        if not traces:
            continue

        n_samples += 1
        method_label = _infer_method_label(traces, method)
        final_meta = traces[-1].get("metadata") or {}
        algorithm_mode = str(
            final_meta.get("algorithm_execution_mode") or method_label
        ).lower()

        ttft = _ttft_from_metadata(final_meta, method_label)
        if ttft is not None:
            ttft_list.append(ttft)

        for trace in traces:
            meta = trace.get("metadata") or {}
            if meta.get("num_hidden_layers"):
                num_layers = int(meta["num_hidden_layers"])
            if meta.get("relay_layer_st") is not None:
                start_layer = int(meta["relay_layer_st"])
            if meta.get("relay_layer_ed") is not None:
                end_layer = int(meta["relay_layer_ed"])

            ph = int(meta.get("ph_token_nums") or 0)
            if ph == 0 and algorithm_mode not in ("relaycaching", "cacheblend"):
                ph = int(meta.get("total_prefix_tokens") or 0)
            acc_ph += ph

            if algorithm_mode == "kvcomm":
                if meta.get("kvcomm_reuse"):
                    kvcomm_reuse += ph
            elif algorithm_mode in ("relaycaching", "cacheblend"):
                acc_recompute += int(meta.get("recompute_num") or 0)
            elif algorithm_mode == "epic":
                acc_epic += int(meta.get("epic_token") or 0)

            total_generation_tokens += int(meta.get("generation_tokens") or 0)

    summary = data[-1] if isinstance(data[-1], dict) else {}
    accuracy = summary.get("Accuracy")
    total_executed = summary.get("Total executed")
    total_solved = summary.get("Total solved")

    reuse_rate = _sample_reuse_rate(
        method_label=method_label,
        algorithm_mode=algorithm_mode,
        acc_ph=acc_ph,
        acc_recompute=acc_recompute,
        acc_epic=acc_epic,
        kvcomm_reuse=kvcomm_reuse,
        num_layers=num_layers,
        start_layer=start_layer,
        end_layer=end_layer,
    )

    return {
        "file": str(json_path.resolve()),
        "method_label": method_label,
        "algorithm_execution_mode": algorithm_mode,
        "accuracy": accuracy,
        "total_executed": total_executed,
        "total_solved": total_solved,
        "num_samples": n_samples,
        "num_ttft_samples": len(ttft_list),
        "ttft_mean_s": _mean(ttft_list),
        "ttft_std_s": _std(ttft_list),
        "ttft_median_s": _median(ttft_list),
        "ttft_min_s": min(ttft_list) if ttft_list else 0.0,
        "ttft_max_s": max(ttft_list) if ttft_list else 0.0,
        "reuse_rate": reuse_rate,
        "ph_tokens": acc_ph,
        "recompute_tokens": acc_recompute,
        "epic_tokens": acc_epic,
        "total_generation_tokens": total_generation_tokens,
        "num_hidden_layers": num_layers,
        "relay_layer_start": start_layer,
        "relay_layer_end": end_layer,
    }


def _print_report(metrics: Dict[str, Any]) -> None:
    ttft_ms = metrics["ttft_mean_s"] * 1000.0
    ttft_std_ms = metrics["ttft_std_s"] * 1000.0
    print(f"File:     {metrics['file']}")
    print(f"Method:   {metrics['method_label']} (algo={metrics['algorithm_execution_mode']})")
    if metrics.get("accuracy") is not None:
        print(
            f"Accuracy: {metrics['accuracy']} "
            f"({metrics.get('total_solved')}/{metrics.get('total_executed')})"
        )
    print(f"Samples:  {metrics['num_samples']} (TTFT n={metrics['num_ttft_samples']})")
    print(f"TTFT:     mean={ttft_ms:.2f} ms  std={ttft_std_ms:.2f} ms  median={metrics['ttft_median_s']*1000:.2f} ms")
    print(f"Reuse:    {metrics['reuse_rate']:.4f}  (ph={metrics['ph_tokens']}, recompute={metrics['recompute_tokens']})")
    print(f"Gen toks: {metrics['total_generation_tokens']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path, help="Experiment result JSON file.")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Override method/baseline label (default: metadata.baseline_label).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON instead of a human-readable report.",
    )
    args = parser.parse_args()

    if not args.json_path.is_file():
        parser.error(f"Not a file: {args.json_path}")

    metrics = analyze_result_json(args.json_path, method=args.method)
    if args.json:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        _print_report(metrics)


if __name__ == "__main__":
    main()
