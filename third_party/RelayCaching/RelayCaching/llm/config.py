from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, Optional
import os


@dataclass(frozen=True)
class RelayCachingConfig:
    """
    Configuration for KV communication and scheduling.

    Args:
        threshold (float): The threshold for the KV communication.
        thread_pool_workers (int): The number of threads to use for the thread pool.
        worker_timeout (float): The timeout for the worker.
    """

    #  Baseline method parameters
    execution_mode: Optional[str] = None  # 'kvcomm', 'relaycaching', 'cacheblend', 'epic'

    # kvcomm baseline (anchor) parameters
    threshold: float = 0.3
    max_anchor_num: int = 20
    window_size: int = 5
    thread_pool_workers: int = 8
    worker_timeout: float = 30.0
    
    # RelayCache parameters
    recompute_label: bool = True
    start_layer: int = 0        # L_start: Hidden states stored/loaded here for reuse
    end_layer: int = -1         # L_end: Selective recomputation stops after this layer
    det_layer: int = 0          # L_det: Detection layer where token deviation is evaluated
    tau_dev: float = 0.0        # tau_dev: Deviation score threshold factor
    tau_inf: float = 0.0        # tau_inf: Influence score threshold factor
    prefix_tokens: int = 0      # Number of prefix tokens to be corrected
    suffix_tokens: int = 0      # Number of suffix tokens to be corrected

    # CacheBlend
    cacheblend_diff_por: float = 0.2      
    cacheblend_select_layer: int = 1    

    # EPIC
    # EPIC baseline prefix token count:
    #   0        → Direct Reuse (RoPE Corrected): reuse KV cache without re-prefilling
    #   1000000  → Full Prefill: re-prefill all tokens (value exceeds any realistic seq length)
    #   16       → EPIC
    epic_prefix_token_num: int = 16 
    

    # Generation parameters
    temperature: float = 1.0
    do_sample: bool = False
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    max_tokens: int = 512

    @classmethod
    def from_env(cls) -> "RelayCachingConfig":
        """Create a config from environment variables with safe defaults."""
        return cls(
            threshold=float(os.environ.get("THRESHOLD", cls.threshold)),
            max_anchor_num=int(os.environ.get("MAX_ANCHOR_NUM", cls.max_anchor_num)),
            window_size=int(os.environ.get("WINDOW_SIZE", cls.window_size)),
            thread_pool_workers=int(os.environ.get("RELAYCACHING_THREAD_WORKERS", cls.thread_pool_workers)),
            worker_timeout=float(os.environ.get("RELAYCACHING_WORKER_TIMEOUT", cls.worker_timeout)),
        ).validate()

    def apply_overrides(self, **overrides: Any) -> "RelayCachingConfig":
        """Return a copy with provided non-None fields overridden."""
        current: Dict[str, Any] = asdict(self)
        for key, value in overrides.items():
            if value is None or key not in current:
                continue
            current[key] = value
        return replace(self, **current).validate()

    def validate(self) -> "RelayCachingConfig":
        """Validate value ranges and return self."""
        if self.thread_pool_workers <= 0:
            raise ValueError("thread_pool_workers must be positive")
        if self.worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        return self

