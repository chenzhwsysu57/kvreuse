from __future__ import annotations

import json
from dataclasses import dataclass, field
import copy
from threading import Lock
from typing import Any, Dict, Optional

from loguru import logger


@dataclass(slots=True)
class GenerationResult:
    """Container describing a single model generation outcome."""

    text: str
    mode: str
    ttft: float
    raw_output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_ttft_for_analysis(
    *,
    execution_mode: str,
    generation_ttft: float,
    preprocess_latency: float = 0.0,
    kvcomm_breakdown_timing: Optional[Dict[str, float]] = None,
) -> float:
    """TTFT used by greedy / paper-style analysis scripts (see README § TTFT reporting).

    RelayCaching / CacheBlend / EPIC / dense_prefill_only
        Returned value = ``generation_ttft`` (wall time until first new token inside ``generate()``).

    KVCOMM
        Returned value = ``generation_ttft`` + pre-``generate`` overhead.
        Prefer ``preprocess_latency`` when present (anchor matching, KV merge before ``generate()``).
        Otherwise fall back to the sum of ``kvcomm_breakdown_timing`` sub-steps from trace
        metadata (KVCOMM runs record task_prefill / anchor_matching / kv_correction).

    Trace metadata mirrors this: non-KVCOMM modes set ``ttft_s == generation_ttft``;
    KVCOMM sets ``ttft_s == generation_ttft + preprocess_latency``.
    """
    base = max(0.0, float(generation_ttft))
    mode = (execution_mode or "").lower()
    if mode != "kvcomm":
        return base
    prep = max(0.0, float(preprocess_latency))
    if prep > 0.0:
        return base + prep
    if kvcomm_breakdown_timing:
        extra = (
            float(kvcomm_breakdown_timing.get("task_prefill_s", 0.0) or 0.0)
            + float(kvcomm_breakdown_timing.get("anchor_matching_s", 0.0) or 0.0)
            + float(kvcomm_breakdown_timing.get("kv_correction_s", 0.0) or 0.0)
        )
        return base + max(0.0, extra)
    return base


def compute_reuse_rate_cost(
    *,
    execution_mode: str,
    ph_tokens: int,
    recompute_tokens: int = 0,
    num_layers: int = 32,
    start_layer: int = 1,
    end_layer: int = 18,
    cacheblend_diff_por: float = 0.2,
    epic_prefix_token_num: int = 0,
    kvcomm_reuse: bool = False,
) -> float:
    """Layer-cost reuse rate (aligned with ``experiments/analysis/analyze_result.py``).

    ``execution_mode`` must be one of ``RelayCaching.utils.utils.MODE_LIST``
    (``relaycaching``, ``kvcomm``, ``cacheblend``, ``epic``, ``dense_prefill_only``).
    Legacy greedy labels such as ``zero`` / ``full`` are not valid here.
    """
    mode = (execution_mode or "").lower()
    ph = max(0, int(ph_tokens))
    if ph <= 0:
        return 0.0
    L = max(1, int(num_layers))
    st = max(0, int(start_layer))
    ed = int(end_layer)
    if ed < 0:
        ed = L - 1
    ed = min(ed, L - 1)

    if mode in ("dense_prefill_only", "dense_prefill"):
        # Full-prefill baseline: no cross-agent KV reuse.
        return 0.0
    if mode in ("epic", "epic16"):
        reused = max(0, ph - int(epic_prefix_token_num))
        return max(0.0, min(1.0, reused / ph))
    if mode == "kvcomm":
        return 1.0 if kvcomm_reuse else 0.0
    if mode == "cacheblend":
        recompute_ratio = max(0.0, float(recompute_tokens)) / ph
        cost = (1.0 / L) + (((L - 1) / L) * recompute_ratio)
        return max(0.0, min(1.0, 1.0 - cost))
    if mode == "relaycaching":
        recompute_ratio = max(0.0, float(recompute_tokens)) / ph
        span = max(0, ed - st - 2)
        cost = (2.0 / L) + ((span / L) * recompute_ratio)
        return max(0.0, min(1.0, 1.0 - cost))
    return 0.0


def _infer_algorithm_mode(agents: list) -> str:
    """Infer algorithm mode from agent metadata (aligned with analyze_result.py)."""
    for entry in agents:
        meta = entry.get("metadata") or {}
        algo = meta.get("algorithm_execution_mode")
        if algo:
            return str(algo).lower()
        label = meta.get("baseline_label")
        if label:
            return str(label).lower()
    return "unknown"


def _aggregate_layer_cost_inputs(
    agents: list,
    *,
    algorithm_mode: str,
) -> Dict[str, Any]:
    """Sum per-agent metadata for reuse-rate calculation (aligned with analyze_result.py)."""
    mode = (algorithm_mode or "unknown").lower()
    acc_ph = 0
    acc_recompute = 0
    acc_epic = 0
    kvcomm_reuse_ph = 0
    num_layers = 32
    start_layer = 1
    end_layer = 18

    for entry in agents:
        meta = entry.get("metadata") or {}
        if meta.get("num_hidden_layers"):
            num_layers = int(meta["num_hidden_layers"])
        if meta.get("relay_layer_st") is not None:
            start_layer = int(meta["relay_layer_st"])
        if meta.get("relay_layer_ed") is not None:
            end_layer = int(meta["relay_layer_ed"])

        ph = int(meta.get("ph_token_nums") or 0)
        if ph == 0 and mode not in ("relaycaching", "cacheblend"):
            ph = int(meta.get("total_prefix_tokens") or 0)
        acc_ph += max(0, ph)

        if mode == "kvcomm":
            if meta.get("kvcomm_reuse"):
                kvcomm_reuse_ph += ph
        elif mode in ("relaycaching", "cacheblend"):
            acc_recompute += int(meta.get("recompute_num") or 0)
        elif mode == "epic":
            acc_epic += int(meta.get("epic_token") or 0)

    return {
        "acc_ph": acc_ph,
        "acc_recompute": acc_recompute,
        "acc_epic": acc_epic,
        "kvcomm_reuse_ph": kvcomm_reuse_ph,
        "num_layers": num_layers,
        "start_layer": start_layer,
        "end_layer": end_layer,
    }


def _layer_cost_reuse_from_totals(
    *,
    execution_mode: str,
    acc_ph: int,
    acc_recompute: int,
    acc_epic: int,
    kvcomm_reuse_ph: int,
    num_layers: int,
    start_layer: int,
    end_layer: int,
    epic_prefix_token_num: int = 0,
) -> float:
    mode = (execution_mode or "unknown").lower()
    if mode == "kvcomm":
        return (kvcomm_reuse_ph / acc_ph) if acc_ph > 0 else 0.0
    if mode == "epic":
        if acc_ph <= 0:
            return 0.0
        return max(0.0, min(1.0, (acc_ph - acc_epic) / acc_ph))
    return compute_reuse_rate_cost(
        execution_mode=mode,
        ph_tokens=acc_ph,
        recompute_tokens=acc_recompute,
        num_layers=num_layers,
        start_layer=start_layer,
        end_layer=end_layer,
        epic_prefix_token_num=epic_prefix_token_num,
        kvcomm_reuse=kvcomm_reuse_ph > 0,
    )


class RequestMetricsRecorder:
    """Tracks per-request agent outputs, reuse rates, and mode-specific TTFT."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._total_agent_calls: int = 0
        self._kv_reuse_agent_calls: int = 0
        self._ttft_stats: Dict[str, Dict[str, float]] = {}
        # Cumulative reuse-rate inputs (paper-style; see analyze_result.py)
        self._acc_ph: int = 0
        self._acc_recompute: int = 0
        self._acc_epic: int = 0
        self._kvcomm_reuse_ph: int = 0
        self._cumulative_algorithm_mode: str = "unknown"
        self._num_layers: int = 32
        self._start_layer: int = 1
        self._end_layer: int = 18

    def start_request(
        self,
        *,
        request_uid: str,
        batch_index: Optional[int],
        task: Optional[str],
        execution_mode: str,
    ) -> None:
        """Initialise bookkeeping for a new request."""
        with self._lock:
            self._requests[request_uid] = {
                "batch_index": batch_index,
                "task": task,
                "execution_mode": execution_mode,
                "agents": [],
                "kv_reuse_agent_calls": 0,
                "total_count": 0,
            }

    def record_agent_output(
        self,
        *,
        request_uid: str,
        agent_id: str,
        agent_name: str,
        agent_role: str,
        generation: GenerationResult | None,
    ) -> None:
        """Log an agent's generation output and update reuse statistics."""
        if generation is None:
            return

        with self._lock:
            request_entry = self._requests.setdefault(
                request_uid,
                {
                    "batch_index": None,
                    "task": None,
                    "execution_mode": "unknown",
                    "agents": [],
                    "kv_reuse_agent_calls": 0,
                    "total_count": 0,
                },
            )

            agent_record = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "ttft": generation.ttft,
                "text": generation.text,
            }
            if generation.metadata:
                agent_record["metadata"] = generation.metadata
            agents_list = request_entry["agents"]
            replaced = False
            for idx, existing in enumerate(agents_list):
                if existing["agent_id"] == agent_id:
                    agents_list[idx] = agent_record
                    replaced = True
                    break
            if not replaced:
                agents_list.append(agent_record)

            request_entry["total_count"] = len(agents_list)
            request_entry["kv_reuse_agent_calls"] = sum(
                1 for entry in agents_list if entry.get("mode") == "kv_reuse"
            )

            stats = self._ttft_stats.setdefault(
                generation.mode, {"sum": 0.0, "count": 0.0}
            )
            stats["sum"] += generation.ttft
            stats["count"] += 1
            avg_ttft = stats["sum"] / stats["count"] if stats["count"] else 0.0

            ttft_payload = {
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "ttft": generation.ttft,
                "mode_avg_ttft": avg_ttft,
            }
            metadata = generation.metadata or {}
            preprocess_latency = metadata.get("preprocess_latency")
            if preprocess_latency is not None:
                ttft_payload["preprocess_latency"] = preprocess_latency
            generation_ttft = metadata.get("generation_ttft")
            if generation_ttft is not None:
                ttft_payload["generation_ttft"] = generation_ttft
            logger.opt(colors=True).info(
                "<cyan>[TTFT:{mode}]</cyan> {}",
                json.dumps(ttft_payload, ensure_ascii=False),
                mode=generation.mode,
            )

            output_payload = {
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "text": generation.text,
            }
            logger.opt(colors=True).info(
                "<green>[AGENT OUTPUT]</green> {}",
                json.dumps(output_payload, ensure_ascii=False),
            )

    def finalize_request(self, request_uid: str) -> Optional[float]:
        """Compute and log per-request reuse statistics."""
        with self._lock:
            request_entry = self._requests.pop(request_uid, None)
            if request_entry is None:
                return None

            agents = request_entry.get("agents", [])
            kv_reuse_agent_calls = int(
                request_entry.get("kv_reuse_agent_calls", 0)
            )
            total_agent_calls = int(request_entry.get("total_count", 0))
            algorithm_mode = _infer_algorithm_mode(agents)
            layer_inputs = _aggregate_layer_cost_inputs(
                agents,
                algorithm_mode=algorithm_mode,
            )
            reuse_rate = _layer_cost_reuse_from_totals(
                execution_mode=algorithm_mode,
                **layer_inputs,
            )

            payload = {
                "request_uid": request_uid,
                "batch_index": request_entry.get("batch_index"),
                "task": request_entry.get("task"),
                "algorithm_execution_mode": algorithm_mode,
                "reuse_rate": round(reuse_rate, 4),
                "kv_reuse_agent_calls": kv_reuse_agent_calls,
                "total_agent_calls": total_agent_calls,
                "ph_tokens": layer_inputs["acc_ph"],
                "recompute_tokens": layer_inputs["acc_recompute"],
            }
            logger.opt(colors=True).info(
                "<magenta>[REQUEST REUSE]</magenta> {}",
                json.dumps(payload, ensure_ascii=False),
            )

            self._total_agent_calls += total_agent_calls
            self._kv_reuse_agent_calls += kv_reuse_agent_calls
            self._cumulative_algorithm_mode = algorithm_mode
            self._acc_ph += layer_inputs["acc_ph"]
            self._acc_recompute += layer_inputs["acc_recompute"]
            self._acc_epic += layer_inputs["acc_epic"]
            self._kvcomm_reuse_ph += layer_inputs["kvcomm_reuse_ph"]
            self._num_layers = layer_inputs["num_layers"]
            self._start_layer = layer_inputs["start_layer"]
            self._end_layer = layer_inputs["end_layer"]
            return reuse_rate

    def log_cumulative(self, *, batch_index: Optional[int]) -> float:
        """Log cumulative reuse rate after a batch (paper / analyze_result.py)."""
        with self._lock:
            reuse_rate = _layer_cost_reuse_from_totals(
                execution_mode=self._cumulative_algorithm_mode,
                acc_ph=self._acc_ph,
                acc_recompute=self._acc_recompute,
                acc_epic=self._acc_epic,
                kvcomm_reuse_ph=self._kvcomm_reuse_ph,
                num_layers=self._num_layers,
                start_layer=self._start_layer,
                end_layer=self._end_layer,
            )

            payload = {
                "batch_index": batch_index,
                "algorithm_execution_mode": self._cumulative_algorithm_mode,
                "reuse_rate": round(reuse_rate, 4),
                "ph_tokens": self._acc_ph,
                "recompute_tokens": self._acc_recompute,
                "kv_reuse_agent_calls": self._kv_reuse_agent_calls,
                "total_agent_calls": self._total_agent_calls,
            }
            summary = (
                f"reuse rate {reuse_rate:.1%} "
                f"(ph={self._acc_ph}, recompute={self._acc_recompute}); "
                f"kv_reuse agents {self._kv_reuse_agent_calls}/{self._total_agent_calls}"
            )
            logger.opt(colors=True).info(
                "<yellow>[CUMULATIVE REUSE]</yellow> {} | {}",
                summary,
                json.dumps(payload, ensure_ascii=False),
            )
            return reuse_rate

    def snapshot_request(self, request_uid: str) -> Optional[Dict[str, Any]]:
        """Return a deep copy of the current request entry without removing it."""
        with self._lock:
            entry = self._requests.get(request_uid)
            if entry is None:
                return None
            return copy.deepcopy(entry)


metrics_recorder = RequestMetricsRecorder()
