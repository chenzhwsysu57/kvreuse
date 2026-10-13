from typing import List, Optional, Tuple, Iterable, Dict
import math

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging

logger = logging.get_logger(__name__)

__all__ = [
    "RelayCache",
    "calc_keep_indices",
    "get_mask_by_mean_threshold",
    "get_topk_by_mean_threshold",
]



def _prepare_scores(scores: torch.Tensor):
    """Sanitize scores for curvature-based selection: nan/inf/neg cleanup."""

    # Sanitize: replace nan/inf with 0, clamp negatives to 0.
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = torch.clamp(scores, min=0.0)
    return scores

def get_mask_by_mean_threshold(scores: torch.Tensor, threshold: float):
    """
    Return a boolean mask selecting tokens above the curvature cutoff.

    This is the fused version that avoids both sorting AND topk:
    instead of  get_optimal_k  -> topk(scores, k)  [O(N) + O(N log K)],
    we directly produce the mask in O(N).
    """
    if not isinstance(scores, torch.Tensor):
        return torch.zeros(0, dtype=torch.bool)

    scores = _prepare_scores(scores)

    num_total = scores.numel()
    if num_total == 0:
        return torch.zeros(0, device=scores.device, dtype=torch.bool)
    if threshold <= 1e-6:
        return torch.ones(num_total, device=scores.device, dtype=torch.bool)

    total = scores.sum()
    if total < 1e-8:
        return torch.zeros(num_total, device=scores.device, dtype=torch.bool)

    cutoff = (total / num_total) * threshold
    mask = scores > cutoff

    return mask

def get_topk_by_mean_threshold(scores: torch.Tensor, threshold: float):
    """
    Return the optimal cutoff count K via mean-based thresholding.

    Delegates to get_mask_by_mean_threshold and returns the count of
    selected tokens.  O(N) with no sorting.
    """
    mask = get_mask_by_mean_threshold(scores, threshold)
    return int(mask.sum().item()) if mask.numel() > 0 else 0




def _fused_deviation_select(
    selected_value: torch.Tensor, 
    old_value: torch.Tensor, 
    tau_dev: float, 
    cached_indices: torch.Tensor, 
    relay_indices: torch.Tensor,
):
    """
    Fused score calculation, top-k filtering, and index generation.
    """
    num_total = selected_value.shape[-2]
    
    # Compute cos_sim once for all subsequent operations
    cos_sim = F.cosine_similarity(selected_value, old_value, dim=-1)  # [B, H, N]
    
    # Mean reduction: [B, H, N] -> [N]
    scores = (1.0 - cos_sim).mean(dim=(0, 1))
    
    # Compute overall deviation (to decide if selection is needed)
    sum_diff = 1.0 - cos_sim.mean()
    
    # Fast short-circuit: deviation too small, no selection needed
    if sum_diff < 1e-6:
        return (
            torch.arange(num_total, device=scores.device, dtype=torch.long),  # top_inner (keep all)
            torch.tensor([], device=scores.device, dtype=torch.long),  # drop_inner
            torch.tensor([], device=scores.device, dtype=torch.long),  # drop_relative
            sum_diff,
        )
    
    # O(N) fused mask: skip both sort and topk
    keep_mask = get_mask_by_mean_threshold(scores, threshold=tau_dev)
    
    if not keep_mask.any():
        # No token selected (extreme case)
        drop_relative = relay_indices if relay_indices.numel() > 0 else torch.tensor([], device=scores.device, dtype=torch.long)
        return (
            torch.tensor([], device=scores.device, dtype=torch.long),
            torch.arange(num_total, device=scores.device, dtype=torch.long),
            drop_relative,
            sum_diff,
        )
    
    # Merge cached_indices into keep_mask
    if cached_indices.numel() > 0:
        keep_mask.index_fill_(0, cached_indices, True)
    
    # === Derive top_inner and drop_inner via double nonzero ===
    top_inner = keep_mask.nonzero(as_tuple=False).squeeze(1)
    drop_inner = (~keep_mask).nonzero(as_tuple=False).squeeze(1)
    
    # Build drop_relative from drop_inner
    if relay_indices.numel() > 0 and drop_inner.numel() > 0:
        drop_relative = relay_indices[drop_inner]
    else:
        drop_relative = torch.tensor([], device=scores.device, dtype=torch.long)
        
    return top_inner, drop_inner, drop_relative, sum_diff

def calc_keep_indices(
    chunk_starts: torch.Tensor,
    chunk_ends: torch.Tensor,
    top_p_tensor: torch.Tensor,
    flat_candidates: torch.Tensor,
    candidate_lens: torch.Tensor,
    prefix_num: int,
    suffix_num: int,
):
    """
    Calculate the global indices of tokens to keep.

    For each chunk, combines:
      - top-p proportional selection from candidates
      - fixed prefix tokens
      - fixed suffix tokens
    Returns a sorted, deduplicated 1-D tensor of global indices.
    """
    # === 1. Basic chunk info ===
    chunk_lens = chunk_ends - chunk_starts + 1
    num_chunks = chunk_lens.size(0)
    device = chunk_starts.device
    
    cum_lens = torch.cumsum(chunk_lens, dim=0)
    # Optimized: use roll instead of pad + slice
    chunk_relative_starts = torch.zeros_like(cum_lens)
    if num_chunks > 1:
        chunk_relative_starts[1:] = cum_lens[:-1]

    # Simplified base count computation
    base_raw_count = prefix_num + suffix_num
    base_occupied = torch.clamp(chunk_lens, max=base_raw_count)
    
    # Target total retention count (Target Budget)
    target_total = (chunk_lens.float() * top_p_tensor).ceil().long()
    
    # Count needed from additional indices
    needed_counts = torch.clamp(target_total - base_occupied, min=0)
    needed_counts = torch.minimum(needed_counts, candidate_lens)

    # === 2. Select additional indices ===
    if flat_candidates.numel() > 0:
        # Pre-compute all needed index mappings
        c_ids_for_candidates = torch.repeat_interleave(
            torch.arange(num_chunks, device=device), 
            candidate_lens
        )
        
        cand_cum_lens = torch.cumsum(candidate_lens, dim=0)
        cand_starts = torch.zeros_like(cand_cum_lens)
        if num_chunks > 1:
            cand_starts[1:] = cand_cum_lens[:-1]
        
        candidate_ranks = torch.arange(flat_candidates.numel(), device=device) - cand_starts[c_ids_for_candidates]
        
        thresholds = needed_counts[c_ids_for_candidates]
        picks_mask = candidate_ranks < thresholds
        
        selected_offsets = flat_candidates[picks_mask]
        selected_c_ids = c_ids_for_candidates[picks_mask]
        additional_global = selected_offsets + chunk_relative_starts[selected_c_ids]
    else:
        additional_global = torch.tensor([], device=device, dtype=torch.long)

    # === 3. Generate base indices ===
    # Merged prefix and suffix generation logic
    
    # Prefix: broadcast + flatten
    if prefix_num > 0:
        p_range = torch.arange(prefix_num, device=device)
        prefix_global = (chunk_relative_starts.unsqueeze(1) + p_range).view(-1)
        c_limits = chunk_relative_starts + chunk_lens
        # Expand limits using repeat_interleave
        expanded_limits = c_limits.repeat_interleave(prefix_num)
        p_valid_mask = prefix_global < expanded_limits
        prefix_global = prefix_global[p_valid_mask]
    else:
        prefix_global = torch.tensor([], device=device, dtype=torch.long)

    # Suffix
    if suffix_num > 0:
        s_range = torch.arange(suffix_num, device=device)
        s_start_offsets = torch.clamp(chunk_lens - suffix_num, min=0)
        s_global_starts = chunk_relative_starts + s_start_offsets
        suffix_global = (s_global_starts.unsqueeze(1) + s_range).view(-1)
        
        c_limits = chunk_relative_starts + chunk_lens
        expanded_limits = c_limits.repeat_interleave(suffix_num)
        s_valid_mask = suffix_global < expanded_limits
        suffix_global = suffix_global[s_valid_mask]
    else:
        suffix_global = torch.tensor([], device=device, dtype=torch.long)
    
    # Single cat + unique for final result
    all_parts = [p for p in [additional_global, prefix_global, suffix_global] if p.numel() > 0]
    
    if len(all_parts) == 0:
        return torch.tensor([], device=device, dtype=torch.long)
    elif len(all_parts) == 1:
        return torch.unique(all_parts[0], sorted=True)
    else:
        all_indices = torch.cat(all_parts)
        return torch.unique(all_indices, sorted=True)

# =============================================================================
#  RELAY CACHE IMPLEMENTATION
# =============================================================================

class RelayCache(Cache):
    """
    RelayCaching KV Cache (Paper Section 4).
    
    Manages selective recomputation of KV cache entries during decoding.
    
    Key parameter mapping (Paper Section 4.2):
        store_hidden_layer  → L_start: Layer where hidden states are stored during prefill
        reuse_hidden_layer  → L_start: Layer where stored hidden states are loaded during reuse
        det_layer           → L_det:   Detection layer (token selection via deviation)
        end_layer           → L_end:   End layer (selective recomputation stops here)
        tau_dev             → tau_dev: Deviation score threshold factor
        tau_inf             → tau_inf: Influence score threshold factor
    
    Layer behavior in relay_select():
        [reuse_hidden_layer, det_layer):  Direct KV scatter (no selection)
        det_layer (= L_det):              Token selection via deviation detection
        (det_layer, end_layer]:           Apply selected indices for recomputation
    """
    def __init__(self, full_prefix_cache: DynamicCache, 
                recompute_indices: Optional[List[int]] = None, 
                blank_absolute_indices: Optional[List[int]] = None, blank_relative_indices: Optional[List[int]] = None,
                relay_absolute_indices: Optional[List[int]] = None, relay_relative_indices: Optional[List[int]] = None,
                recompute_label: bool = False,
                det_layer: Optional[int] = None, end_layer: Optional[int] = None, tau_dev: float = 0, 
                relay_chunk_indices: Optional[List[List[int]]] = None, additional_indices: Optional[List[List[int]]] = None, additional_scores: Optional[List[List[float]]] = None,
                reuse_hidden_layer: Optional[int] = None, store_hidden_layer: Optional[int] = None,
                tau_inf: float = 0.5,
                max_length: int = 8192,
                save_prefill_hidden: bool = False,
                profile_upper_layer: bool = False,
                profile_lower_layer: bool = False,
                epic_prefix_token_num: int = 0,
                cacheblend_diff_por: float = 0.0,
                execution_mode: str = "default",
                prefix_tokens: int = 0,
                suffix_tokens: int = 10,
                reusable_hidden_states_list : Optional[List[torch.Tensor]] = None,
                _distributed_cache_data: Optional[Iterable] = None ) -> None:
        # Fallback for mutable default arguments (avoid shared mutable state)
        if recompute_indices is None:
            recompute_indices = []
        if blank_absolute_indices is None:
            blank_absolute_indices = []
        if blank_relative_indices is None:
            blank_relative_indices = []
        if relay_absolute_indices is None:
            relay_absolute_indices = []
        if relay_relative_indices is None:
            relay_relative_indices = []
        if reusable_hidden_states_list is None:
            reusable_hidden_states_list = []
        super().__init__()

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0 
        
        if len(full_prefix_cache.key_cache) > 0 and full_prefix_cache.key_cache[0].shape[-2] > 0:
            batch_size, num_heads, prefix_len, head_dim = full_prefix_cache.key_cache[0].shape
            self._seen_tokens = prefix_len 
            self.head_dim = head_dim
            self.num_layers = len(full_prefix_cache.key_cache)
            
            if prefix_len > max_length:
                self.max_length = prefix_len
            else:
                self.max_length = max_length
            
            self.device = full_prefix_cache.value_cache[0].device
            dtype = full_prefix_cache.value_cache[0].dtype
        else:
            raise ValueError("Prefix cache must contain at least one layer with non-zero length.")
        
        # Pre-allocate cache
        for i in range(self.num_layers):
            new_k = torch.full((batch_size, num_heads, self.max_length, head_dim), -1, dtype=dtype, device=self.device)
            new_v = torch.full((batch_size, num_heads, self.max_length, head_dim), -1, dtype=dtype, device=self.device)

            if i < len(full_prefix_cache.key_cache) and prefix_len > 0:
                new_k[:, :, :prefix_len, :] = full_prefix_cache.key_cache[i]
                new_v[:, :, :prefix_len, :] = full_prefix_cache.value_cache[i]
            
            self.key_cache.append(new_k)
            self.value_cache.append(new_v)
        
        # Recompute-related fields
        self.recompute_label = recompute_label

        def to_tensor(x):
            if isinstance(x, torch.Tensor): return x.to(self.device, dtype=torch.long)
            return torch.tensor(x if x is not None else [], device=self.device, dtype=torch.long)

        self.recompute_indices_tensor = to_tensor(recompute_indices)
        self.blank_absolute_indices_tensor = to_tensor(blank_absolute_indices)
        self.blank_relative_indices_tensor = to_tensor(blank_relative_indices)
        self.relay_absolute_indices_tensor = to_tensor(relay_absolute_indices)
        self.relay_relative_indices_tensor = to_tensor(relay_relative_indices)
        self.relay_chunk_tensor = to_tensor(relay_chunk_indices)  # stores chunk info
        # Record initial relay token count before any selection/pruning
        self._initial_relay_num = int(self.relay_absolute_indices_tensor.numel())
        
        # additional_indices is a List[List[int]], where each sublist corresponds to
        # a chunk's extra indices (irregular and not sequentially ordered)
        self.det_layer = det_layer 
        self.end_layer = end_layer 
        self.tau_dev = tau_dev 
        self.tau_inf = tau_inf 

        self.drop_label_pre = False
        self.drop_label_post = False 

        self.reuse_hidden_layer = reuse_hidden_layer 
        self.store_hidden_layer = store_hidden_layer 
        self.reusable_hidden_states_list = reusable_hidden_states_list 
        self.response_hidden_states_list = []
        self.reusable_hidden_states  = None

        self.save_prefill_hidden = save_prefill_hidden
        self.profile_upper_layer = profile_upper_layer 
        self.profile_lower_layer = profile_lower_layer 

        self.downstream_influence_scores = torch.zeros(
            (self.max_length,), 
            device=self.device, 
            dtype=torch.float32
        )
        self.score_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        # Compatible with HuggingFace Cache distributed serialization interface
        if _distributed_cache_data is not None:
            for key_states, value_states in _distributed_cache_data:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)

        self.epic_prefix_token_num = epic_prefix_token_num
        # cacheblend select ratio
        self.cacheblend_diff_por = cacheblend_diff_por
        # CacheBlend baseline: selection always at layer 1
        self.cacheblend_select_layer = 1
        self.execution_mode = execution_mode
        self.decode_start_idx = None
        self.prefix_tokens = prefix_tokens
        self.suffix_tokens = suffix_tokens
        
        self.set_additional_indices(additional_indices, additional_scores)
    
        self._prepare_reusable_hidden_states()


            
        if self.execution_mode == 'relaycaching' and self.relay_absolute_indices_tensor.numel() and additional_indices is not None and len(additional_indices) > 0:
            # Construct p_list (assuming all chunks use the same tau_dev)
            # If different chunks have different p values, adjust here accordingly
            # relay_chunk_tensor shape: [num_chunks, 2]; use dim0 for chunk count
            num_chunks = self.relay_chunk_tensor.shape[0]
            top_p_tensor = torch.full(
                (num_chunks,), 
                self.tau_inf,
                device=self.device, 
                dtype=torch.float32
            )
            with torch.no_grad():
                self.cached_additional_indices = self._get_additional_inner_relative_indices(top_p_tensor)
        else:
            self.cached_additional_indices = torch.tensor([], device=self.device, dtype=torch.long)
        # `recompute_num` is used by upper-level metrics/logging as "tokens to recompute".
        # - relaycaching: determined later by online selection (top-p)
        # - cacheblend: determined by diff-based selection ratio
        # - epic: fixed policy => recompute the (already-truncated) relay indices
        self.recompute_num = 0
        if self.execution_mode == "epic":
            try:
                self.recompute_num = int(self.relay_absolute_indices_tensor.numel())
            except Exception:
                self.recompute_num = 0
        
        # Pre-initialize all dynamic attributes to avoid hasattr checks
        self.selection_done = False
        self.top_inner_relative = None
        self.drop_inner_relative = None
        self.drop_relative = None
        self.relative_chunk_info = None
        self.deep_layer_similarity_stats = {
            'cos_sim_avg': [],
            'layer_indices': []
        }
        self._arange_cache: Dict[int, torch.Tensor] = {}
        self._early_drop_done = False
        self._skip_initial_drop = False
        self._cached_normal_indices = None
        self._cached_rotary_emb_func = None
        self._load_hidden_buffer = None



    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self):
            return ( self.key_cache[layer_idx][..., :self._seen_tokens, :], 
            self.value_cache[layer_idx][..., :self._seen_tokens, :])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield ( self.key_cache[layer_idx][..., :self._seen_tokens, :], 
            self.value_cache[layer_idx][..., :self._seen_tokens, :])

    def __len__(self):
        return len(self.key_cache)
    
    def is_decoding(self, query_states: torch.Tensor) -> bool:
        """Check if the current step is a decode (single-token) step."""
        return query_states.shape[-2] == 1
    
    def get_recompute_num(self):
        """
        Get the number of tokens that need recomputation.

        Returns:
            int: Number of tokens to recompute.
        """
        return self.recompute_num

    def get_similarity_stats(self):
        """
        Per-layer decode-vs-recompute KV similarity collected during relay selection.

        Returns:
            dict: ``cos_sim_avg`` and ``layer_indices`` when recording is enabled.
        """
        return self.deep_layer_similarity_stats

    def record_deep_layer_similarity(self, layer_idx, selected_value, old_value):
        """
        Record per-layer similarity between decode-time KV and recomputed KV.

        At the relay selection layer, ``old_value`` is the value cache produced
        during normal decode at relay token positions; ``selected_value`` is the
        KV from the current forward pass that fully recomputes those positions.
        When enabled, cosine-sim summaries are appended to
        ``deep_layer_similarity_stats`` (see ``get_similarity_stats()`` /
        ``metadata['layer_sim']``). Disabled (no-op) in the open-source release.
        """
        return
    
    def initial_process(self, hidden_states, causal_mask, position_embeddings, position_ids, rotary_emb_func):
        if self.execution_mode != 'relaycaching':
            return hidden_states, causal_mask, position_embeddings, position_ids
        if self.recompute_label and self.reuse_hidden_layer >= 0 and self.reusable_hidden_states is not None:
            self.ori_position_embeddings = position_embeddings
            self.ori_position_ids = position_ids
            self.ori_causal_mask = causal_mask
            
            # Cache rotary_emb_func for load_hidden_states early-drop path
            self._cached_rotary_emb_func = rotary_emb_func
            
            self.ori_recompute_indices_tensor = self.recompute_indices_tensor
            self.ori_blank_relative_indices_tensor = self.blank_relative_indices_tensor
            self.ori_relay_relative_indices_tensor = self.relay_relative_indices_tensor

            # If _skip_initial_drop is set, the forward pass already
            # embedded only non-relay tokens, so hidden_states is already short.
            _skip_drop = self._skip_initial_drop
            if _skip_drop:
                self._skip_initial_drop = False  # reset flag
                seq_len = self.ori_position_ids.size(-1)
                keep_indices = self._get_keep_indices_fast(self.relay_relative_indices_tensor, seq_len)
                position_ids = self.ori_position_ids[:, keep_indices]
                if causal_mask is not None:
                    causal_mask = causal_mask[:, :, keep_indices, :]
                position_embeddings = rotary_emb_func(hidden_states, position_ids)
            else:
                # Use Scatter Write to obtain keep indices
                seq_len = position_ids.size(-1)
                keep_indices = self._get_keep_indices_fast(self.relay_relative_indices_tensor, seq_len)
                position_ids = position_ids[:, keep_indices]
                hidden_states = hidden_states[:, keep_indices, :]
                if causal_mask is not None:
                    causal_mask = causal_mask[:, :, keep_indices, :]
                position_embeddings = rotary_emb_func(hidden_states, position_ids)
            
            # Cache normal_indices for load_hidden_states to reuse
            self._cached_normal_indices = keep_indices
            
            # Pre-allocate hidden_states merge buffer for load_hidden_states
            batch_size = hidden_states.shape[0]
            total_seq_len = self.ori_position_ids.shape[1]
            hidden_dim = hidden_states.shape[2]
            self._load_hidden_buffer = torch.empty(
                (batch_size, total_seq_len, hidden_dim),
                dtype=hidden_states.dtype, device=self.device
            )
            
            # Assign as Tensor
            self.recompute_indices_tensor = self.blank_absolute_indices_tensor
            
            self.blank_relative_indices_tensor = torch.arange(
                self.recompute_indices_tensor.numel(), device=self.device, dtype=torch.long
            )
            self.relay_relative_indices_tensor = torch.tensor([], device=self.device, dtype=torch.long)
        return hidden_states, causal_mask, position_embeddings, position_ids
    
    def _prepare_reusable_hidden_states(self):
        if len(self.reusable_hidden_states_list) == 0:
            self.reusable_hidden_states = None
        elif self.reusable_hidden_states is None: 
            rhs = torch.cat(
                self.reusable_hidden_states_list, dim=1
            ).contiguous()
            # Pre-align rhs shape to avoid runtime defensive checks in load_hidden_states.
            if rhs.dim() == 2:
                rhs = rhs.unsqueeze(0)  # (L, H) -> (1, L, H)
            # Align length to relay_relative_indices_tensor
            target_len = self.relay_relative_indices_tensor.numel()
            if target_len > 0 and rhs.shape[1] != target_len:
                rep = (target_len + rhs.shape[1] - 1) // rhs.shape[1]
                rhs = rhs.repeat(1, rep, 1)[:, :target_len, :]
            self.reusable_hidden_states = rhs
            self.reusable_hidden_states_list = [] 

    def _expand_cache(self, required_length: int, old_seen_tokens):
        new_max_length = max(self.max_length * 2, required_length)
        
        for i in range(len(self.key_cache)):
            current_k = self.key_cache[i]
            current_v = self.value_cache[i]
            batch_size, num_heads, _, head_dim = current_k.shape
            
            new_k = torch.full((batch_size, num_heads, new_max_length, head_dim), -1, 
                               dtype=current_k.dtype, device=current_k.device)
            new_v = torch.full((batch_size, num_heads, new_max_length, head_dim), -1, 
                               dtype=current_v.dtype, device=current_v.device)
            
            valid_len = old_seen_tokens
            new_k[..., :valid_len, :] = current_k[..., :valid_len, :]
            new_v[..., :valid_len, :] = current_v[..., :valid_len, :]
            
            self.key_cache[i] = new_k
            self.value_cache[i] = new_v
        
        if self.downstream_influence_scores is not None:
            new_scores = torch.zeros((new_max_length,), device=self.device, dtype=torch.float32)
            new_scores[:self.max_length] = self.downstream_influence_scores
            self.downstream_influence_scores = new_scores

        self.max_length = new_max_length

    def load_hidden_states(self, hidden_states, layer_idx, causal_mask, position_embeddings, position_ids):
        if self.execution_mode != 'relaycaching':
            return hidden_states, causal_mask, position_embeddings, position_ids
        if (layer_idx == self.reuse_hidden_layer and self.reusable_hidden_states is not None): 
            batch_size, _, hidden_dim = hidden_states.shape

            # Early-drop: when selection is done, only restore selected relay tokens
            # instead of all relay tokens, reducing sequence length for subsequent layers.
            _use_early_drop = (
                self.selection_done
                and self.top_inner_relative is not None
                and isinstance(self.top_inner_relative, torch.Tensor)
                and self.top_inner_relative.numel() > 0
                and self.top_inner_relative.numel() < self.ori_relay_relative_indices_tensor.numel()
            )

            if _use_early_drop:
                # --- Early-drop path: only restore selected relay tokens ---
                # selected_relay_relative: positions of selected relay tokens in the ORIGINAL full sequence
                selected_inner = self.top_inner_relative  # inner indices within relay tokens
                selected_relay_relative = self.ori_relay_relative_indices_tensor[selected_inner]

                # normal_indices: positions of non-relay tokens in the original full sequence
                normal_indices = self._cached_normal_indices
                if normal_indices is None:
                    full_seq_len = self.ori_position_ids.shape[1]
                    normal_indices = self._get_keep_indices_fast(
                        self.ori_relay_relative_indices_tensor, full_seq_len
                    )

                # Build the reduced sequence: [selected_relay + normal] sorted by position
                keep_positions = torch.cat([selected_relay_relative, normal_indices])
                keep_positions, sort_order = torch.sort(keep_positions)

                # Build reduced hidden_states by concatenating relay hidden states + normal hidden states
                rhs = self.reusable_hidden_states
                if rhs is not None:
                    if rhs.dtype != hidden_states.dtype:
                        rhs = rhs.to(dtype=hidden_states.dtype)
                    if rhs.shape[0] == 1 and batch_size > 1:
                        rhs = rhs.expand(batch_size, -1, -1)
                # rhs has shape [B, all_relay_count, H]. Select only the kept ones.
                selected_rhs = rhs[:, selected_inner, :]  # [B, n_keep, H]

                # Concatenate: [selected_relay_hs, normal_hs] then reorder by sort_order
                merged_hs = torch.cat([selected_rhs, hidden_states], dim=1)  # [B, n_keep + n_normal, H]
                merged_hs = merged_hs[:, sort_order, :]  # reorder to match keep_positions order

                hidden_states = merged_hs

                # Build reduced position_ids, causal_mask, position_embeddings
                ori_pos_ids = self.ori_position_ids
                position_ids = ori_pos_ids[:, keep_positions]

                ori_causal = self.ori_causal_mask
                if ori_causal is not None:
                    causal_mask = ori_causal[:, :, keep_positions, :]
                else:
                    causal_mask = None

                # Recompute position_embeddings for the reduced sequence
                # (We need a rotary_emb_func but don't have it here. Store it from initial_process.)
                _rotary_fn = self._cached_rotary_emb_func
                if _rotary_fn is not None:
                    position_embeddings = _rotary_fn(hidden_states, position_ids)
                else:
                    # Fallback: slice from original position_embeddings
                    ori_pe = self.ori_position_embeddings
                    if ori_pe is not None and isinstance(ori_pe, tuple) and len(ori_pe) == 2:
                        cos, sin = ori_pe
                        if cos.dim() == 3:  # [B, S, D]
                            position_embeddings = (cos[:, keep_positions, :], sin[:, keep_positions, :])
                        elif cos.dim() == 2:  # [S, D]
                            position_embeddings = (cos[keep_positions, :], sin[keep_positions, :])
                        else:
                            position_embeddings = ori_pe
                    else:
                        position_embeddings = ori_pe

                # Update relay state: now relay_absolute_indices only contains selected ones
                selected_relay_absolute = self.relay_absolute_indices_tensor[selected_inner]
                # Rebuild recompute_indices as sorted(selected_relay_absolute + blank_absolute)
                if self.blank_absolute_indices_tensor.numel() > 0:
                    merged_abs = torch.cat([selected_relay_absolute, self.blank_absolute_indices_tensor])
                else:
                    merged_abs = selected_relay_absolute
                self.recompute_indices_tensor, _ = torch.sort(merged_abs)

                # Compute new relative indices
                self.relay_absolute_indices_tensor = selected_relay_absolute
                if selected_relay_absolute.numel() > 0:
                    self.relay_relative_indices_tensor = torch.searchsorted(
                        self.recompute_indices_tensor, selected_relay_absolute
                    )
                else:
                    self.relay_relative_indices_tensor = torch.tensor([], device=self.device, dtype=torch.long)

                if self.blank_absolute_indices_tensor.numel() > 0:
                    self.blank_relative_indices_tensor = torch.searchsorted(
                        self.recompute_indices_tensor, self.blank_absolute_indices_tensor
                    )
                else:
                    self.blank_relative_indices_tensor = torch.arange(0, device=self.device, dtype=torch.long)

                # Mark that pre_drop at change_layer+1 should be skipped
                # (we already have the right sequence length)
                self._early_drop_done = True
                # Clear drop_relative since the drop has already been done.
                self.drop_relative = torch.tensor([], device=self.device, dtype=torch.long)
                self.drop_inner_relative = torch.tensor([], device=self.device, dtype=torch.long)
                # Update top_inner_relative to be identity (all relay tokens are now "top")
                self.top_inner_relative = torch.arange(
                    selected_relay_absolute.numel(), device=self.device, dtype=torch.long
                )
                # Also update ori_* to reflect the reduced sequence so that
                # relay_select at deeper layers uses correct indices
                self.ori_recompute_indices_tensor = self.recompute_indices_tensor
                self.ori_relay_relative_indices_tensor = self.relay_relative_indices_tensor
                self.ori_blank_relative_indices_tensor = self.blank_relative_indices_tensor

                self.reusable_hidden_states = None

            else:
                # --- Original path: restore ALL relay tokens ---
                total_seq_len = self.ori_position_ids.shape[1] 
                
                # Reuse pre-allocated buffer from initial_process if available;
                # fall back to torch.empty only when buffer is missing or shape mismatches.
                buf = self._load_hidden_buffer
                if (buf is not None
                        and buf.shape[0] == batch_size
                        and buf.shape[1] == total_seq_len
                        and buf.shape[2] == hidden_dim
                        and buf.dtype == hidden_states.dtype):
                    new_hidden_states = buf
                else:
                    new_hidden_states = torch.empty(
                        (batch_size, total_seq_len, hidden_dim),
                        dtype=hidden_states.dtype, device=self.device
                    )

                # Reuse cached normal_indices from initial_process;
                # fall back to recomputation only when cache is missing.
                normal_indices = self._cached_normal_indices
                if normal_indices is None:
                    normal_indices = self._get_keep_indices_fast(
                        self.ori_relay_relative_indices_tensor, total_seq_len
                    )
                
                relay_indices = self.ori_relay_relative_indices_tensor
                
                # Scatter write: fill Normal positions with current hidden_states
                new_hidden_states[:, normal_indices, :] = hidden_states
                
                # rhs is pre-aligned in _prepare_reusable_hidden_states;
                # only need lightweight dtype cast and batch expansion here.
                rhs = self.reusable_hidden_states
                if rhs is not None:
                    if rhs.dtype != hidden_states.dtype:
                        rhs = rhs.to(dtype=hidden_states.dtype)
                    if rhs.shape[0] == 1 and batch_size > 1:
                        rhs = rhs.expand(batch_size, -1, -1)
                new_hidden_states[:, relay_indices, :] = rhs
                
                hidden_states = new_hidden_states
                
                # Restore original state
                position_ids = self.ori_position_ids
                position_embeddings = self.ori_position_embeddings
                causal_mask = self.ori_causal_mask
                
                self.recompute_indices_tensor = self.ori_recompute_indices_tensor
                self.blank_relative_indices_tensor = self.ori_blank_relative_indices_tensor
                self.relay_relative_indices_tensor = self.ori_relay_relative_indices_tensor
                self.reusable_hidden_states = None

        return hidden_states, causal_mask, position_embeddings, position_ids
    
    def save_hidden_states(self, hidden_states, layer_idx):
        if self.execution_mode != 'relaycaching':
            return
        if layer_idx == self.store_hidden_layer:
            if hidden_states.shape[-2] == 1:
                self.response_hidden_states_list.append(hidden_states)
                # reuse_hidden_layer should always equal store_hidden_layer;
                self.reuse_hidden_layer = self.store_hidden_layer
            elif self.save_prefill_hidden:
                self.response_hidden_states_list.append(hidden_states)
                self.store_hidden_layer = None 

    def pre_drop_process(self, hidden_states, causal_mask, position_embeddings, position_ids, rotary_emb_func):
        # If early-drop was done in load_hidden_states, skip pre_drop
        if self._early_drop_done and self.recompute_label and self.drop_label_pre:
            self.drop_label_pre = False
            self._early_drop_done = False
            return hidden_states, causal_mask, position_embeddings, position_ids
        if self.recompute_label and self.drop_label_pre:
            seq_len = hidden_states.shape[1]
            
            # Get keep indices (complement of drop_relative)
            keep_indices = self._get_keep_indices_fast(self.drop_relative, seq_len)

            # Integer indexing for position_ids and hidden_states
            position_ids = position_ids[:, keep_indices]
            hidden_states = hidden_states[:, keep_indices, :]
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, keep_indices, :]

            position_embeddings = rotary_emb_func(hidden_states, position_ids)

            # Full GPU pipeline: filter relay indices via tensor indexing
            self.relay_absolute_indices_tensor = self.relay_absolute_indices_tensor[self.top_inner_relative]

            merged_indices = torch.cat([self.relay_absolute_indices_tensor, self.blank_absolute_indices_tensor])
            self.recompute_indices_tensor, _ = torch.sort(merged_indices)

            # Compute blank positions in new_recompute using GPU searchsorted
            self.blank_relative_indices_tensor = torch.searchsorted(
                self.recompute_indices_tensor, 
                self.blank_absolute_indices_tensor
            )
            
            if self.relay_absolute_indices_tensor.numel() > 0:
                self.relay_relative_indices_tensor = torch.searchsorted(
                    self.recompute_indices_tensor,
                    self.relay_absolute_indices_tensor
                )
            else:
                self.relay_relative_indices_tensor = torch.tensor([], device=self.device, dtype=torch.long)
                
            self.drop_label_pre = False

        return hidden_states, causal_mask, position_embeddings, position_ids

    def post_drop_process(self, hidden_states, causal_mask, position_embeddings, position_ids, rotary_emb_func):
        if self.recompute_label and self.drop_label_post:
            num_recompute = self.recompute_indices_tensor.numel()
            
            idx_blank = self.blank_relative_indices_tensor
            idx_normal = torch.arange(
                start=num_recompute, end=hidden_states.shape[-2], device=self.device, dtype=torch.long
            )
            keep_indices = torch.cat([idx_blank, idx_normal])
            
            hidden_states = hidden_states[:, keep_indices, :]
            position_ids = position_ids[:, keep_indices]
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, keep_indices, :]

            position_embeddings = rotary_emb_func(hidden_states, position_ids)

            self.relay_absolute_indices_tensor = torch.tensor([], device=self.device, dtype=torch.long)
            self.relay_relative_indices_tensor = torch.tensor([], device=self.device, dtype=torch.long)

            # Reset recompute to blank (note: Tensor assignment here)
            self.recompute_indices_tensor = self.blank_absolute_indices_tensor

            # Reset blank relative to 0...N
            self.blank_relative_indices_tensor = torch.arange(
                self.recompute_indices_tensor.numel(), device=self.device, dtype=torch.long
            )   
            self.drop_label_post = False
            self.recompute_label = False

        return hidden_states, causal_mask, position_embeddings, position_ids

    def update_cache(self, key_states, value_states, layer_idx, cache_kwargs=None):
        num_recompute = self.recompute_indices_tensor.numel()
        selected_key = key_states[..., :num_recompute, :] 
        selected_value = value_states[..., :num_recompute, :]
        new_key_states = key_states[..., num_recompute:, :] 
        new_value_states = value_states[..., num_recompute:, :]
        
        if self.relay_absolute_indices_tensor.numel() > 0 and self.relay_relative_indices_tensor.numel() > 0:
            relay_selected_key = selected_key[..., self.relay_relative_indices_tensor, :]
            relay_selected_value = selected_value[..., self.relay_relative_indices_tensor, :]
            
            if self.profile_lower_layer:
                self.relay_select_for_profile_lower(layer_idx, relay_selected_key, relay_selected_value)
            elif self.profile_upper_layer:
                self.relay_select_for_profile_upper(layer_idx, relay_selected_key, relay_selected_value)
            elif self.execution_mode == 'cacheblend':
                self.cacheblend_select(layer_idx, relay_selected_key, relay_selected_value)
            elif self.execution_mode == 'epic':
                self.epic_select(layer_idx, relay_selected_key, relay_selected_value)
            else:
                self.relay_select(layer_idx, relay_selected_key, relay_selected_value)

        if self.blank_absolute_indices_tensor.numel() > 0:
            blank_selected_key = selected_key[..., self.blank_relative_indices_tensor, :]
            blank_selected_value = selected_value[..., self.blank_relative_indices_tensor, :]
            self.key_cache[layer_idx][..., self.blank_absolute_indices_tensor, :] = blank_selected_key
            self.value_cache[layer_idx][..., self.blank_absolute_indices_tensor, :] = blank_selected_value
    
        num_new_tokens = new_key_states.shape[-2]
        if num_new_tokens == 0:
            return
        old_seen_tokens = self._seen_tokens
        if layer_idx == 0:
            st_idx  = self._seen_tokens
            end_idx = self._seen_tokens + num_new_tokens
            self._seen_tokens = end_idx
        else:
            st_idx = self._seen_tokens - num_new_tokens
            end_idx = self._seen_tokens
        
        if end_idx > self.max_length:
            self._expand_cache(end_idx, old_seen_tokens)
        
        self.key_cache[layer_idx][..., st_idx:end_idx, :] = new_key_states
        self.value_cache[layer_idx][..., st_idx:end_idx, :] = new_value_states
   
        
    def _compute_influence_scores(self, query_states, key_states_slice):
        """
        Shared core logic: GQA expansion + attention score computation.

        Returns the raw attention probabilities tensor for accumulation.
        """
        H_k = key_states_slice.shape[1]
        H_q = query_states.shape[1]

        if H_q != H_k:
            if H_q % H_k != 0:
                raise ValueError(f"Query heads ({H_q}) must be divisible by Key heads ({H_k}).")
            num_groups = H_q // H_k
            # Use repeat_interleave instead of expand + reshape
            # This avoids creating large intermediate tensors
            key_states_expanded = key_states_slice.repeat_interleave(num_groups, dim=1)
        else:
            key_states_expanded = key_states_slice

        # Compute attention scores directly, avoid creating full attention matrix
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', query_states, key_states_expanded) * scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        return attn_probs

    def _calculate_and_add_scores(self, query_states, key_states_slice, target_end_idx):
        """
        Compute and accumulate downstream influence scores.

        Uses in-place accumulation with reduced memory allocation for GQA expansion.
        """
        attn_probs = self._compute_influence_scores(query_states, key_states_slice)
        scores_to_add = attn_probs.sum(dim=(0, 1, 2))
        self.downstream_influence_scores[:target_end_idx].add_(scores_to_add)

    def _calculate_and_add_scores_async(self, query_states, key_states_slice, target_end_idx):
        """
        Async version of downstream score computation.

        Uses a separate CUDA stream for computation without blocking the main flow.
        """
        if self.score_stream is None:
            # No available stream, fall back to synchronous version
            self._calculate_and_add_scores(query_states, key_states_slice, target_end_idx)
            return

        # Get current stream
        current_stream = torch.cuda.current_stream()

        # Execute computation on score_stream
        with torch.cuda.stream(self.score_stream):
            # Wait for main stream to finish data preparation
            self.score_stream.wait_stream(current_stream)

            # record_stream tells PyTorch these tensors are still in use in score_stream
            query_states.record_stream(self.score_stream)
            key_states_slice.record_stream(self.score_stream)

            # Perform the actual computation
            attn_probs = self._compute_influence_scores(query_states, key_states_slice)
            scores_to_add = attn_probs.sum(dim=(0, 1, 2))
            self.downstream_influence_scores[:target_end_idx].add_(scores_to_add)
    
    def _sync_async_scores(self):
        """
        Synchronize all async score computations.
        Call this before using score results.
        """
        if self.score_stream is not None:
            self.score_stream.synchronize()
    
    def update_decode(self, query_states, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.decode_start_idx is None:
            self.decode_start_idx = self._seen_tokens
        self.update_cache(key_states, value_states, layer_idx, cache_kwargs)
        
        if self.execution_mode != 'relaycaching':
            return (self.key_cache[layer_idx][..., :self._seen_tokens, :], 
                    self.value_cache[layer_idx][..., :self._seen_tokens, :])
        
        if self.head_dim is None:
            self.head_dim = query_states.shape[-1]
            
        current_total_len = self._seen_tokens 
        decode_start = self.decode_start_idx
        slice_start = decode_start
        slice_end = current_total_len
        
        current_k_view = self.key_cache[layer_idx][..., slice_start:slice_end, :]
        
        self._calculate_and_add_scores_async(query_states, current_k_view, current_total_len - decode_start)
        
        return (self.key_cache[layer_idx][..., :self._seen_tokens, :], 
                self.value_cache[layer_idx][..., :self._seen_tokens, :])


    def get_ranked_influence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._sync_async_scores()
        scores = self.downstream_influence_scores[:self._seen_tokens]
        sorted_scores, relative_sorted_indices = torch.sort(scores, descending=True)
        return sorted_scores, relative_sorted_indices
    
    def update_prefill(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.update_cache(key_states, value_states, layer_idx, cache_kwargs)
        return (self.key_cache[layer_idx][..., :self._seen_tokens, :], 
                self.value_cache[layer_idx][..., :self._seen_tokens, :])
    
   
    def relay_select_for_profile_upper(self, layer_idx, selected_key, selected_value):
        current_indices_tensor = self.relay_absolute_indices_tensor
        if (layer_idx == self.reuse_hidden_layer):
            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value
        elif layer_idx == self.reuse_hidden_layer + 1:
            # Prepare parameters (no Dynamic flags needed, handled by Wrapper)
            old_value = self.value_cache[layer_idx][..., current_indices_tensor, :]
            
            cached_indices = self.cached_additional_indices 
            relay_indices = self.relay_relative_indices_tensor
            self.top_inner_relative, self.drop_inner_relative, self.drop_relative, _ = _fused_deviation_select(
                selected_value, 
                old_value, 
                self.tau_dev,
                cached_indices, 
                relay_indices,
            )
            # Write back to Cache (using original indices is fine, no need for dynamic behavior)
            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value
            self.drop_label_pre = True
        elif layer_idx < self.det_layer and layer_idx > self.reuse_hidden_layer + 1:
            # Use pre-initialized properties instead of hasattr check
            assert self.top_inner_relative is not None, "Top-p indices not found for deep layers."
            
            # Perform the update
            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value


    def relay_select_for_profile_lower(self, layer_idx, selected_key, selected_value):
        current_indices_tensor = self.relay_absolute_indices_tensor
        if (layer_idx >= self.det_layer - 1):
            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value

    def cacheblend_select(self, layer_idx, selected_key: torch.Tensor, selected_value: torch.Tensor):
        current_indices_tensor = self.relay_absolute_indices_tensor
        num_current_indices = self.relay_relative_indices_tensor.numel()
        keep_ratio = self.cacheblend_diff_por
        select_layer = 1

        if num_current_indices == 0:
            self.recompute_num = 0
            if layer_idx >= select_layer:
                self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
                self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value
            return

        # Always write KV for the current layer first/anyway.
        # For selection layer, we additionally compute diff/topk and enable drop for the next layers.
        if layer_idx < select_layer:
            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value
            return
        
        if layer_idx == select_layer:
            # If selection is already done, skip diff-based selection and just write KV,
            # then enable drop for subsequent layers.
            if self.selection_done:
                self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
                self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value
                self.drop_label_pre = True
                return

            old_value = self.value_cache[layer_idx][..., current_indices_tensor, :]
            diff_score = torch.norm(selected_value - old_value, p=2, dim=-1)
            token_diff_score = diff_score.mean(dim=(0, 1)) 

            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value

            num_to_keep = max(1, int(num_current_indices * keep_ratio))
            _, sorted_indices = torch.sort(token_diff_score, descending=True)
            self.recompute_num = num_to_keep
            self.top_inner_relative = sorted_indices[:num_to_keep]
            self.top_inner_relative, _ = torch.sort(self.top_inner_relative)

            all_inner_indices = torch.arange(num_current_indices, device=self.device)
            drop_mask = ~torch.isin(all_inner_indices, self.top_inner_relative)
            self.drop_inner_relative = all_inner_indices[drop_mask]
            self.drop_relative = self.relay_relative_indices_tensor[self.drop_inner_relative]
            self.drop_label_pre = True
            return

        else:
            # layer_idx > select_layer: write back KV for the (already-pruned) relay indices
            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value
    
    def epic_select(self, layer_idx, selected_key: torch.Tensor, selected_value: torch.Tensor):
        current_indices_tensor = self.relay_absolute_indices_tensor
        self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
        self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value

    def set_additional_indices(self, indices: List[List[int]], scores: Optional[List[List[float]]] = None):
        """
        Call once during data loading or config update.
        Complete all CPU -> GPU conversions here.
        """
        self.additional_indices = indices
        if indices:
            # Flatten operation done only once here
            self.additional_indices_flat = torch.tensor(
                [x for sub in indices for x in sub], 
                device=self.device, dtype=torch.long
            )
            self.additional_indices_lens = torch.tensor(
                [len(sub) for sub in indices], 
                device=self.device, dtype=torch.long
            )
            # Flatten scores to GPU tensor
            if scores:
                self.additional_scores_flat = torch.tensor(
                    [s for sub in scores for s in sub],
                    device=self.device, dtype=torch.float32
                )
            else:
                # If no scores provided, default to all 1.0 (equivalent to uniform selection)
                self.additional_scores_flat = torch.ones_like(
                    self.additional_indices_flat, dtype=torch.float32
                )
        else:
            self.additional_indices_flat = torch.tensor([], device=self.device, dtype=torch.long)
            self.additional_indices_lens = torch.tensor([], device=self.device, dtype=torch.long)
            self.additional_scores_flat = torch.tensor([], device=self.device, dtype=torch.float32)

    def _get_additional_inner_relative_indices(self, top_p_for_each_ctx_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get additional inner relative indices.
        
        Caches score_chunks split results and vectorizes proportion computation for each chunk.
        """
        if self.relay_chunk_tensor.numel() == 0 or self.additional_indices_flat.numel() == 0:
            return torch.tensor([], device=self.device, dtype=torch.long)

        num_chunks = self.relay_chunk_tensor.shape[0]
        
        # Vectorized proportion computation for each chunk
        # Vectorized curvature: avoid Python loop over chunks
        # For single-chunk case (most common), skip split entirely
        if num_chunks == 1:
            chunk_scores = self.additional_scores_flat
            if chunk_scores.numel() == 0:
                top_p_tensor = torch.zeros(1, device=self.device, dtype=torch.float32)
            else:
                opt_k = get_topk_by_mean_threshold(chunk_scores, threshold=self.tau_inf)
                chunk_p = opt_k / max(1, chunk_scores.numel())
                top_p_tensor = torch.tensor([chunk_p], device=self.device, dtype=torch.float32)
        else:
            lens_list = self.additional_indices_lens.tolist()
            score_chunks = torch.split(self.additional_scores_flat, lens_list)
            auto_p_list = []
            for i in range(num_chunks):
                chunk_scores = score_chunks[i]
                if chunk_scores.numel() == 0:
                    auto_p_list.append(0.0)
                    continue
                opt_k = get_topk_by_mean_threshold(chunk_scores, threshold=self.tau_inf)
                chunk_p = opt_k / max(1, chunk_scores.numel())
                auto_p_list.append(chunk_p)
            top_p_tensor = torch.tensor(auto_p_list, device=self.device, dtype=torch.float32)

        chunk_starts = self.relay_chunk_tensor[:, 0].contiguous()
        chunk_ends = self.relay_chunk_tensor[:, 1].contiguous()

        return calc_keep_indices(
            chunk_starts, 
            chunk_ends, 
            top_p_tensor,
            self.additional_indices_flat,
            self.additional_indices_lens,
            self.prefix_tokens,
            self.suffix_tokens
        )


    def _get_keep_indices_fast(self, drop_indices: torch.Tensor, total_len: int) -> torch.Tensor:
        """
        Fast computation of keep indices (complement of drop_indices).
        
        Caches commonly used arange results and uses efficient mask operations.
        """
        if drop_indices.numel() == 0:
            if total_len not in self._arange_cache:
                self._arange_cache[total_len] = torch.arange(
                    total_len, device=self.device, dtype=torch.long
                )
            return self._arange_cache[total_len]
        
        # Use scatter operation directly
        mask = torch.ones(total_len, device=self.device, dtype=torch.bool)
        mask.scatter_(0, drop_indices, False)
        return mask.nonzero(as_tuple=False).squeeze(1)
    
    def relay_select(self, layer_idx, selected_key: torch.Tensor, selected_value: torch.Tensor):
        current_indices_tensor = self.relay_absolute_indices_tensor
        # Fast check
        if current_indices_tensor.numel() == 0:
            return

        change_layer = self.det_layer - 1 if self.det_layer is not None else -1
        reuse_hidden_layer = self.reuse_hidden_layer if self.reuse_hidden_layer is not None else change_layer
            
        # Shallow layer direct assignment, skip unnecessary old_value read
        if (reuse_hidden_layer <= layer_idx <= change_layer):
            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value
        # Selection already done; at change_layer+1, still write KV and enable drop_label_pre,
        # even though selection was pre-computed. Without this, change_layer+1 is
        # skipped entirely (branch 2 requires `not selection_done`, branch 3
        # requires `layer_idx > change_layer + 1`), so drop never triggers.
        elif self.selection_done and layer_idx == change_layer + 1:
            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value
            self.drop_label_pre = True
        # Core selection layer logic
        elif not self.selection_done and layer_idx >= change_layer + 1:

            # Prepare parameters (no Dynamic flags needed, handled by Wrapper)
            old_value = self.value_cache[layer_idx][..., current_indices_tensor, :]

            # Runtime KV similarity (decode cache vs. recomputed); disabled in open-source:
            # self.record_deep_layer_similarity(layer_idx, selected_value, old_value)

            cached_indices = self.cached_additional_indices 
            relay_indices = self.relay_relative_indices_tensor

            # Use pre-initialized properties check
            if self.relative_chunk_info is None:
                # This step only needs to be computed once and cached
                chunk_lens = self.relay_chunk_tensor[:, 1] - self.relay_chunk_tensor[:, 0] + 1
                cum_lens = torch.cumsum(chunk_lens, dim=0)
                starts = torch.zeros_like(cum_lens)
                if cum_lens.numel() > 1:
                    starts[1:] = cum_lens[:-1]
                ends = cum_lens - 1
                self.relative_chunk_info = torch.stack([starts, ends], dim=1)

            # Online high-diff token selection
            self.top_inner_relative, self.drop_inner_relative, self.drop_relative, avg_diff = _fused_deviation_select(
                    selected_value, 
                    old_value, 
                    self.tau_dev,
                    cached_indices, 
                    relay_indices,
                )
            self.recompute_num = self.top_inner_relative.numel()
            # Selection layer: all relay tokens were fully recomputed, so write back
            # the complete KV. Dropped tokens are only skipped in *deep* layers;
            # at this layer their KV is still needed during decode-time attention.
            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value
            if avg_diff < 1e-6:
                logger.debug(
                    "Layer {}: CosSim deviation {:.6f} too small, deferring selection to next layer.",
                    layer_idx,
                    avg_diff,
                )
            else:
                self.selection_done = True
                self.drop_label_pre = True
        
        # Deep layer logic
        elif (self.selection_done and layer_idx > change_layer + 1 and
            layer_idx <= self.end_layer):
            # Use pre-initialized properties instead of hasattr check
            assert self.top_inner_relative is not None, "Top-p indices not found for deep layers."
            self.key_cache[layer_idx][..., current_indices_tensor, :] = selected_key
            self.value_cache[layer_idx][..., current_indices_tensor, :] = selected_value
        
        if layer_idx == self.end_layer:
            self.drop_label_post = True

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        is_empty_layer = (len(self.key_cache) == 0 or len(self.key_cache) <= layer_idx or not self.key_cache[layer_idx].numel())
        layer_seq_length = self._seen_tokens if not is_empty_layer else 0
        return layer_seq_length


