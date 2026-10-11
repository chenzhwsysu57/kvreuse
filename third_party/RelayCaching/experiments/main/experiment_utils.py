"""
Shared utilities for RelayCaching main experiments.

This module extracts common logic from individual experiment runners
(run_gsm8k.py, run_humaneval.py, run_mmlu.py, run_aime24.py) to reduce
code duplication and improve maintainability.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch

from RelayCaching.llm.config import RelayCachingConfig
from RelayCaching.utils.globals import Time
from RelayCaching.utils.log import configure_logging, logger
from RelayCaching.utils.utils import MODE_LIST, resolve_baseline_label
from RelayCaching.utils.profiler_utils import seed_everything


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

def setup_cuda_early():
    """Parse --cuda-devices from sys.argv before torch is imported."""
    cuda_flag = "--cuda-devices"
    cuda_value = None
    for idx, arg in enumerate(sys.argv):
        if arg.startswith(f"{cuda_flag}="):
            cuda_value = arg.split("=", 1)[1]
            break
        if arg == cuda_flag and idx + 1 < len(sys.argv):
            cuda_value = sys.argv[idx + 1]
            break
    if cuda_value:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_value


def setup_environment(seed: int = 42):
    """Set random seeds and configure stdout encoding."""
    sys.stdout.reconfigure(encoding="utf-8")
    seed_everything(seed)


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def load_result(result_file: Path) -> list:
    """Load existing results from a JSON file, creating it if necessary."""
    if not result_file.exists() or os.path.getsize(result_file) == 0:
        os.makedirs(result_file.parent, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump([], f)
        return []
    try:
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning(f"Result file {result_file} is corrupted, starting fresh.")
        return []


def save_result(data: list, result_file: Path):
    """Save results to a JSON file."""
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Agent Trace Formatting
# ---------------------------------------------------------------------------

def format_agent_traces(trace: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format raw graph trace into a structured list of agent outputs."""
    if not trace:
        return []
    formatted = []
    for agent in trace.get("agents", []):
        metadata = agent.get("metadata") or {}
        formatted.append({
            "agent_id": agent.get("agent_id"),
            "agent_name": agent.get("agent_name"),
            "agent_role": agent.get("agent_role"),
            "mode": agent.get("mode"),
            "ttft": agent.get("ttft"),
            "input": metadata.get("prompt"),
            "output": agent.get("text"),
            "metadata": metadata,
        })
    return formatted


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def dataloader(data_list: list, batch_size: int, i_batch: int) -> list:
    """Get a batch of data from the dataset."""
    return data_list[i_batch * batch_size: (i_batch + 1) * batch_size]


# ---------------------------------------------------------------------------
# Run Tag Generation
# ---------------------------------------------------------------------------

def _sanitize_component(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        parts = [_sanitize_component(item) for item in value]
        return "-".join(part for part in parts if part) or "none"
    if value is None:
        return "none"
    text = str(value).strip()
    if not text:
        return "none"
    text = text.replace(os.sep, "-").replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("-_.")
    return text or "none"


def _short_component(value: Any, max_len: int = 32) -> str:
    text = _sanitize_component(value)
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def build_run_tag(args: argparse.Namespace, current_time: str) -> str:
    """Generate a descriptive run tag for output file naming."""
    agent_names = "-".join(args.agent_names)
    agent_nums = "-".join(str(num) for num in args.agent_nums)
    llm_short = args.llm_name.split("/")[-1]
    kv_parts = [
        str(value)
        for value in (args.kv_threshold, args.kv_max_anchor_num, args.kv_window_size)
        if value is not None
    ]
    sections = [
        current_time,
        f"mode-{_short_component(args.mode, 20)}",
        f"agents-{_short_component(agent_names, 30)}",
        f"nums-{_short_component(agent_nums, 20)}",
        f"llm-{_short_component(llm_short, 30)}",
        f"exec-{_short_component(args.execution_mode, 20)}",
    ]
    if kv_parts:
        sections.append(f"kv-{_short_component('-'.join(kv_parts), 25)}")
    sections.append(
        f"det-{_short_component(args.det_layer, 10)}-end-{_short_component(args.end_layer, 10)}"
    )
    sections.append(f"taudev-{_short_component(args.tau_dev, 10)}")
    sections.append(f"start-{_short_component(args.start_layer, 10)}")
    sections.append(f"prefix-{_short_component(args.prefix_tokens, 10)}")
    sections.append(f"suffix-{_short_component(args.suffix_tokens, 10)}")
    sections.append(f"tauinf-{_short_component(args.tau_inf, 10)}")
    sections.append(f"epic-{_short_component(args.epic_prefix_token_num, 10)}")
    return "__".join(sections)


# ---------------------------------------------------------------------------
# RelayCachingConfig Construction
# ---------------------------------------------------------------------------

def build_relaycaching_config(args: argparse.Namespace) -> Optional[RelayCachingConfig]:
    """
    Build RelayCachingConfig based on the execution_mode argument.

    Supported modes:
      - kvcomm: Similarity-based KV retrieval (KVCOMM baseline)
      - relaycaching: Our method with selective rectification
      - cacheblend: Value-norm-based detection (CacheBlend baseline)
      - epic: Static prefix recomputation (EPIC baseline)
      - dense_prefill_only: Full prefill without KV reuse (reference)
    """
    sampling_kwargs = dict(
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    # TTFT: only kvcomm adds preprocess_latency to reported ttft; other modes use generation_ttft only.
    if args.execution_mode == "kvcomm":
        return RelayCachingConfig.from_env().apply_overrides(
            threshold=args.kv_threshold,
            max_anchor_num=args.kv_max_anchor_num,
            window_size=args.kv_window_size,
            thread_pool_workers=args.kv_thread_workers,
            worker_timeout=args.kv_worker_timeout,
            execution_mode="kvcomm",
            **sampling_kwargs,
        )
    elif args.execution_mode == "relaycaching":
        return RelayCachingConfig.from_env().apply_overrides(
            det_layer=args.det_layer,
            end_layer=args.end_layer,
            tau_dev=args.tau_dev,
            start_layer=args.start_layer,
            execution_mode="relaycaching",
            tau_inf=args.tau_inf,
            prefix_tokens=args.prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            **sampling_kwargs,
        )
    elif args.execution_mode == "cacheblend":
        assert 0.0 <= args.cacheblend_diff_por <= 1.0, \
            "cacheblend_diff_por must be in [0.0, 1.0]"
        return RelayCachingConfig.from_env().apply_overrides(
            cacheblend_diff_por=args.cacheblend_diff_por,
            execution_mode="cacheblend",
            **sampling_kwargs,
        )
    elif args.execution_mode == "epic":
        assert args.epic_prefix_token_num >= 0, \
            "epic_prefix_token_num must be non-negative"
        return RelayCachingConfig.from_env().apply_overrides(
            epic_prefix_token_num=args.epic_prefix_token_num,
            execution_mode="epic",
            **sampling_kwargs,
        )
    elif args.execution_mode == "dense_prefill_only":
        return RelayCachingConfig.from_env().apply_overrides(
            execution_mode="dense_prefill_only",
            **sampling_kwargs,
        )
    else:
        return RelayCachingConfig.from_env()


# ---------------------------------------------------------------------------
# Common Argument Parser
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared across all experiment runners."""
    # Model
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model name or local path.")

    # Topology
    parser.add_argument("--mode", type=str, default="FullConnected",
                        choices=["DirectAnswer", "FullConnected", "Random",
                                 "Chain", "Debate", "Layered", "Star", "Mesh"],
                        help="Communication topology among agents.")
    parser.add_argument("--agent_names", nargs="+", type=str, default=["MathSolver"],
                        help="List of agent names in the graph.")
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[3],
                        help="List of agent counts corresponding to agent names.")
    parser.add_argument("--decision_method", type=str, default="FinalRefer",
                        help="Decision method for the graph.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of samples per batch.")

    # Execution mode
    parser.add_argument("--execution_mode", type=str, default="relaycaching",
                        choices=["default", "relaycaching", "kvcomm",
                                 "cacheblend", "epic", "dense_prefill_only",
                                 "allow_kv_reuse"],
                        help="KV reuse strategy.")

    # RelayCaching parameters (Paper Section 4)
    # Parameter mapping (Paper → CLI → Config field):
    #   L_start (hidden states layer)       → --start_layer   → config.start_layer (direct)
    #   L_det   (detection/selection layer)  → --det_layer     → config.det_layer (direct)
    #   L_end   (recomputation end layer)   → --end_layer     → config.end_layer (direct)
    # All parameter names are now directly aligned with paper notation.
    #   Layers [start_layer, det_layer) do direct KV assignment (no selection).
    parser.add_argument("--start_layer", type=int, default=1,
                        help="L_start: hidden states stored/loaded here for reuse (Paper Section 4.2). Must be < det_layer.")
    parser.add_argument("--det_layer", type=int, default=3,
                        help="L_det: detection/selection layer where token deviation is evaluated (Paper Section 4.2).")
    parser.add_argument("--end_layer", type=int, default=18,
                        help="L_end: selective recomputation stops here (Paper Section 4.2).")
    parser.add_argument("--tau_dev", type=float, default=1.5,
                        help="Deviation score threshold factor tau_dev.")
    parser.add_argument("--tau_inf", type=float, default=1.45,
                        help="Influence score threshold factor tau_inf.")
    parser.add_argument("--prefix_tokens", type=int, default=0,
                        help="Number of prefix boundary tokens to always recompute.")
    parser.add_argument("--suffix_tokens", type=int, default=10,
                        help="Number of suffix boundary tokens to always recompute.")

    # KVCOMM baseline parameters
    parser.add_argument("--kv-threshold", type=float, default=0.3,
                        help="Similarity threshold for KVCOMM anchor matching.")
    parser.add_argument("--kv-max-anchor-num", type=int, default=20,
                        help="Maximum number of anchors for KVCOMM.")
    parser.add_argument("--kv-window-size", type=int, default=5,
                        help="Window size for KVCOMM anchor update.")
    parser.add_argument("--kv-thread-workers", type=int, default=10,
                        help="Thread workers for KVCOMM processing.")
    parser.add_argument("--kv-worker-timeout", type=float, default=None,
                        help="Timeout for KVCOMM workers.")

    # CacheBlend baseline parameters
    parser.add_argument("--cacheblend_diff_por", type=float, default=0.2,
                        help="Fixed recomputation ratio for CacheBlend.")

    # EPIC baseline parameters
    parser.add_argument("--epic_prefix_token_num", type=int, default=0,
                        help="Number of prefix tokens for EPIC recomputation.")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save experiment results.")
    parser.add_argument("--prefix", type=str, default="Q:\n",
                        help="Prefix text for input query formatting.")

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0, greedy when do_sample=False).")
    parser.add_argument("--do-sample", action="store_true",
                        help="Enable stochastic sampling (default: greedy decoding).")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling probability threshold.")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling threshold.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty factor.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    # Hardware
    parser.add_argument("--cuda-devices", type=str, default=None,
                        help="Comma-separated CUDA device IDs (e.g., '0,1').")
    parser.add_argument("--data_limit", type=int, default=None,
                        help="Limit number of data samples (for quick testing).")


# ---------------------------------------------------------------------------
# Graph Topology (Communication Masks)
# ---------------------------------------------------------------------------

def get_topology_kwargs(
    mode: str,
    num_agents: int,
) -> Dict[str, Any]:
    """
    Generate spatial and temporal masks for the given communication topology.

    Args:
        mode: One of DirectAnswer, FullConnected, Random, Chain, Debate, Layered, Star, Mesh.
        num_agents: Total number of agents in the graph.

    Returns:
        Dictionary with fixed_spatial_masks, fixed_temporal_masks, and node_kwargs.
    """
    N = num_agents
    node_kwargs = None

    if mode == "DirectAnswer":
        spatial = [[0]]
        temporal = [[0]]
        node_kwargs = [{"role": "Programming Expert"}]
    elif mode == "FullConnected":
        spatial = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        temporal = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Random":
        spatial = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        temporal = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == "Chain":
        spatial = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        temporal = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == "Debate":
        spatial = [[0 for _ in range(N)] for _ in range(N)]
        temporal = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Layered":
        spatial = _generate_layered_graph(N)
        temporal = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Star":
        spatial = _generate_star_graph(N)
        temporal = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Mesh":
        spatial = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        temporal = [[1 for _ in range(N)] for _ in range(N)]
    else:
        raise ValueError(f"Unknown topology mode: {mode}")

    return {
        "fixed_spatial_masks": spatial,
        "fixed_temporal_masks": temporal,
        "node_kwargs": node_kwargs,
    }


def _generate_layered_graph(n: int, layer_num: int = 2) -> List[List[int]]:
    """Generate a layered DAG adjacency matrix."""
    adj = [[0] * n for _ in range(n)]
    base_size = n // layer_num
    remainder = n % layer_num
    layers = []
    for i in range(layer_num):
        size = base_size + (1 if i < remainder else 0)
        layers.extend([i] * size)
    random.shuffle(layers)
    for i in range(n):
        for j in range(n):
            if layers[j] == layers[i] + 1:
                adj[i][j] = 1
    return adj


def _generate_star_graph(n: int) -> List[List[int]]:
    """Generate a star topology (hub broadcasts to all)."""
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = 1
    return matrix


# ---------------------------------------------------------------------------
# Experiment Setup Helper
# ---------------------------------------------------------------------------

def setup_experiment(args: argparse.Namespace, domain: str):
    """
    Common experiment setup: seeds, logging, output directory, run tag.

    Returns:
        Tuple of (output_dir, result_file, run_tag, kv_config)
    """
    setup_environment(args.seed)

    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Output directory
    project_root = Path(__file__).resolve().parents[2]
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = project_root / "result" / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    # Time tag
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    # Run tag and logging
    run_tag = build_run_tag(args, current_time)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=logs_dir / f"log.{run_tag}.txt")

    # Result file
    result_file = output_dir / f"{domain}_{run_tag}.json"

    # Algorithm mode (RelayCachingConfig) vs graph scheduling mode (Graph.arun) differ.
    algorithm_mode = args.execution_mode
    args.baseline_label = resolve_baseline_label(
        algorithm_mode,
        epic_prefix_token_num=int(getattr(args, "epic_prefix_token_num", 0) or 0),
    )
    args.algorithm_mode = algorithm_mode

    kv_config = build_relaycaching_config(args)

    # Graph.arun() scheduling: all KV-reuse baselines share this mode string.
    if args.execution_mode in ("kvcomm", "relaycaching", "cacheblend",
                                "epic", "dense_prefill_only"):
        args.execution_mode = "allow_kv_reuse"

    logger.info(
        "RelayCachingConfig: {} | baseline_label={} | graph_mode={}",
        kv_config,
        args.baseline_label,
        args.execution_mode,
    )

    return output_dir, result_file, run_tag, kv_config
