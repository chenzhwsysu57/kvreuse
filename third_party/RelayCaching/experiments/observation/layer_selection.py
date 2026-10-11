"""
Layer Range Auto-Selection Algorithms (Paper Section 4.2)
==========================================================

Implements the automatic layer range and detection layer selection for RelayCaching.
Determines L_start, L_end (based on U-shaped similarity profile) and L_det
(based on Spearman rank correlation stabilization).

Functions:
- auto_select_middle_layer_range: From pre-computed layer similarities → (L_start, L_end)
- auto_select_middle_layer_range_from_kv: From raw KV cache tensors → (L_start, L_end)
- compute_spearman_correlations: Per-sample adjacent-layer Spearman ρ scalars
- merge_average_spearman_correlations: Average ρ across calibration samples (multi-sample L_det)
- auto_select_detection_layer: ρ → α → L_det (single sample or multi-sample via per_sample_token_deviations)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional


def auto_select_middle_layer_range(
    layer_similarities: list,
    lower_threshold: float = 0.99,
    upper_sigma_multiplier: float = 2.0,
    upper_consecutive_count: int = 2,
    upper_min_samples: int = 3,
    right_tail_layers: int = 5,
    metric: str = "value_cosine",
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Select paper-critical layer range [L_start, L_end] (both inclusive; same as CLI bounds).

    ``L_end`` is the last layer in the relay recompute band (same as CLI ``--end_layer``).

    Selection strategy (Paper Sec. 4.2):
    1. L_start / stlayer: deepest layer index with sim >= lower_threshold (τ).
       Layers [0, L_start) reuse KV directly; rectification runs on [L_start, L_end].

    2. Upper bound (L_end): stable-region onset (Paper End layer) from valley scan (λ, C, N_min, T).
       After $\ell_{\min}$, find the first run of $C$ layers with
       $s_\ell \ge \mu_{\text{last}} - \lambda\sigma_{\text{last}}$ and small $|s_\ell - s_{\ell-1}|$;
       $L_{\text{end}}$ is the onset index (left boundary of the deep stable band).

    Args:
        layer_similarities: per-layer similarity list, each element is a dict or float
        lower_threshold: L_start threshold (default 0.99)
        upper_sigma_multiplier: λ, stable-region multiplier (default 2.0)
        upper_consecutive_count: C, consecutive stable layers (default 2)
        upper_min_samples: N_min, min rise samples for Welford (default 3)
        right_tail_layers: T, last layers used to estimate deep stable stats (default 5)
        metric: similarity name ("key_cosine", "value_cosine", "key_norm", or "value_norm")
        verbose: whether to print details

    Returns:
        (L_start, L_end): inclusive critical range (``--start_layer``, ``--end_layer``)
    """
    num_layers = len(layer_similarities)
    
    # Extract the similarity values for the specified metric
    if isinstance(layer_similarities[0], dict):
        similarities = [layer_sims[metric] for layer_sims in layer_similarities]
    else:
        similarities = layer_similarities
    
    if verbose:
        print(f"\n  Auto-selecting middle layer range...")
        print(f"   Metric: {metric}, Threshold: {lower_threshold}")
        print(f"   Total layers: {num_layers}")
    
    # ========== Step 1: Lower Bound (L_start / stlayer) ==========
    last_high_sim_layer = -1
    for layer_idx in range(num_layers):
        if similarities[layer_idx] >= lower_threshold:
            last_high_sim_layer = layer_idx
    lower_bound = last_high_sim_layer if last_high_sim_layer >= 0 else 0

    if verbose:
        print(
            f"   L_start (stlayer): Layer {lower_bound} "
            f"(deepest index with sim >= {lower_threshold}; "
            f"critical range [{lower_bound}, L_end])"
        )
    
    # ========== Step 2: Upper Bound ==========
    # Find the valley (minimum similarity point)
    valley_layer = int(np.argmin(similarities))
    min_sim = similarities[valley_layer]
    
    # Compute statistics for the stable region on the right side (last T layers)
    right_tail_layers = max(1, min(int(right_tail_layers), num_layers))
    right_side_sims = similarities[-right_tail_layers:]
    right_side_mean = np.mean(right_side_sims)
    right_side_std = np.std(right_side_sims)
    
    right_region_lower = right_side_mean - upper_sigma_multiplier * right_side_std
    
    if verbose:
        print(f"   Valley: Layer {valley_layer} (sim={min_sim:.4f})")
        print(
            f"   Right tail T={right_tail_layers}: "
            f"mean={right_side_mean:.4f}, std={right_side_std:.4f}"
        )
        print(f"   Entry threshold: sim >= {right_region_lower:.4f}")
    
    # Traverse from valley toward deeper layers
    upper_bound = num_layers
    n = 0
    rise_mean = 0.0
    M2 = 0.0
    consecutive_stable_count = 0
    stable_start_layer = None
    
    for layer_idx in range(valley_layer + 1, num_layers):
        current_sim = similarities[layer_idx]
        prev_sim = similarities[layer_idx - 1]
        rise = current_sim - prev_sim
        
        # Update via Welford's algorithm
        n += 1
        delta = rise - rise_mean
        rise_mean += delta / n
        delta2 = rise - rise_mean
        M2 += delta * delta2
        
        if n >= upper_min_samples:
            # Check whether we've entered the stable region
            in_right_region = current_sim >= right_region_lower
            change_threshold = max(right_side_std * 2, 0.02)
            change_is_stable = abs(rise) < change_threshold
            is_stable = in_right_region and change_is_stable
            
            if is_stable:
                if consecutive_stable_count == 0:
                    stable_start_layer = layer_idx - 1
                consecutive_stable_count += 1
                
                if consecutive_stable_count >= upper_consecutive_count:
                    upper_bound = stable_start_layer
                    break
            else:
                consecutive_stable_count = 0
                stable_start_layer = None
    
    L_end = min(num_layers - 1, max(lower_bound, upper_bound))
    if upper_bound >= num_layers:
        L_end = num_layers - 1

    if verbose:
        print(f"   L_end (stable onset, inclusive): Layer {L_end}")

    # Validation and adjustment
    if lower_bound > L_end:
        if verbose:
            print(f"   L_start ({lower_bound}) > L_end ({L_end}), using defaults")
        lower_bound = int(num_layers * 0.25)
        L_end = int(num_layers * 0.85) - 1

    min_range = max(3, num_layers // 10)
    if L_end - lower_bound + 1 < min_range:
        if verbose:
            print(f"   Range too small, expanding to {min_range} layers")
        mid = (lower_bound + L_end) // 2
        lower_bound = max(0, mid - min_range // 2)
        L_end = min(num_layers - 1, mid + min_range // 2)

    if verbose:
        n_critical = max(0, L_end - lower_bound + 1)
        print(
            f"   Final range: shallow [0, {max(0, lower_bound - 1)}], "
            f"critical [{lower_bound}, {L_end}] ({n_critical} layers); "
            f"--start_layer {lower_bound} --end_layer {L_end}"
        )

    return lower_bound, L_end


def auto_select_middle_layer_range_from_kv(
    decoding_kv_list: list,
    prefill_kv_list: list,
    num_layers: int,
    lower_threshold: float = 0.99,
    upper_sigma_multiplier: float = 2.0,
    upper_consecutive_count: int = 2,
    upper_min_samples: int = 3,
    right_tail_layers: int = 5,
    metric: str = "value_cosine",
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Automatically select the middle layer range from KV Cache data
    
    First compute layer similarities, then call auto_select_middle_layer_range
    
    Args:
        decoding_kv_list: Decoding KV list (multiple contexts)
        prefill_kv_list: Prefill KV list (multiple contexts)
        num_layers: number of layers
        lower_threshold: L_start threshold
        upper_sigma_multiplier: stable region range multiplier
        upper_consecutive_count: consecutive stable count
        upper_min_samples: minimum sample count
        metric: similarity metric
        verbose: whether to print details
    
    Returns:
        (lower_bound, upper_bound): layer range
    """
    # Compute per-layer similarity (averaged over all contexts)
    layer_similarities = []
    
    for layer_idx in range(num_layers):
        key_cos_sims = []
        value_cos_sims = []
        key_norm_sims = []
        value_norm_sims = []
        
        for decoding_kv, prefill_kv in zip(decoding_kv_list, prefill_kv_list):
            dec_k = decoding_kv[layer_idx]['key']
            dec_v = decoding_kv[layer_idx]['value']
            pre_k = prefill_kv[layer_idx]['key']
            pre_v = prefill_kv[layer_idx]['value']
            
            # Ensure correct dimensions
            if dec_k.dim() == 3:
                dec_k = dec_k.unsqueeze(0)
                dec_v = dec_v.unsqueeze(0)
                pre_k = pre_k.unsqueeze(0)
                pre_v = pre_v.unsqueeze(0)
            
            if dec_k.dim() == 4:
                dec_k, dec_v, pre_k, pre_v = dec_k.squeeze(0), dec_v.squeeze(0), pre_k.squeeze(0), pre_v.squeeze(0)

            head_key_cos, head_val_cos = [], []
            head_key_norm_ratio, head_val_norm_ratio = [], []
            for h in range(dec_k.shape[0]):
                k1, k2 = dec_k[h], pre_k[h]
                v1, v2 = dec_v[h], pre_v[h]
                if k1.shape[0] == 0 or k1.shape != k2.shape:
                    continue
                head_key_cos.append(F.cosine_similarity(k1, k2, dim=1).mean().item())
                head_val_cos.append(F.cosine_similarity(v1, v2, dim=1).mean().item())
                k1_n, k2_n = torch.norm(k1, p=2, dim=1), torch.norm(k2, p=2, dim=1)
                v1_n, v2_n = torch.norm(v1, p=2, dim=1), torch.norm(v2, p=2, dim=1)
                head_key_norm_ratio.append((k2_n / (k1_n + 1e-8)).mean().item())
                head_val_norm_ratio.append((v2_n / (v1_n + 1e-8)).mean().item())

            key_cos = float(np.mean(head_key_cos)) if head_key_cos else 0.0
            value_cos = float(np.mean(head_val_cos)) if head_val_cos else 0.0
            key_norm = float(1.0 - np.mean(np.abs(np.array(head_key_norm_ratio) - 1.0))) if head_key_norm_ratio else 0.0
            value_norm = float(1.0 - np.mean(np.abs(np.array(head_val_norm_ratio) - 1.0))) if head_val_norm_ratio else 0.0
            
            key_cos_sims.append(key_cos)
            value_cos_sims.append(value_cos)
            key_norm_sims.append(key_norm)
            value_norm_sims.append(value_norm)
        
        layer_similarities.append({
            'key_cosine': float(np.mean(key_cos_sims)),
            'value_cosine': float(np.mean(value_cos_sims)),
            'key_norm': float(np.mean(key_norm_sims)),
            'value_norm': float(np.mean(value_norm_sims))
        })
    
    return auto_select_middle_layer_range(
        layer_similarities=layer_similarities,
        lower_threshold=lower_threshold,
        upper_sigma_multiplier=upper_sigma_multiplier,
        upper_consecutive_count=upper_consecutive_count,
        upper_min_samples=upper_min_samples,
        right_tail_layers=right_tail_layers,
        metric=metric,
        verbose=verbose
    )


# L_det: use all high-deviation tokens for adjacent-layer Spearman (not exposed as CLI)
LDET_TOPK_RATIO = 1.0


def compute_spearman_correlations(
    token_deviations: Dict[int, np.ndarray],
    topk_ratio: float = 0.2,
) -> Dict[int, float]:
    """
    Per adjacent layer pair (L, L+1): Spearman rho on high-deviation tokens at layer L.

    Returns {L: rho} keyed by the **pair start** layer L (same as exp_plot x-axis).
    """
    from scipy.stats import spearmanr

    sorted_layers = sorted(token_deviations.keys())
    correlations: Dict[int, float] = {}
    for i in range(len(sorted_layers) - 1):
        layer_l = sorted_layers[i]
        layer_l1 = sorted_layers[i + 1]
        devs_l = token_deviations[layer_l]
        devs_l1 = token_deviations[layer_l1]
        topk = max(2, int(len(devs_l) * topk_ratio))
        topk_indices = np.argsort(devs_l)[-topk:]
        devs_l_topk = devs_l[topk_indices]
        devs_l1_topk = devs_l1[topk_indices]
        if len(topk_indices) > 1 and np.std(devs_l_topk) > 0 and np.std(devs_l1_topk) > 0:
            corr, _ = spearmanr(devs_l_topk, devs_l1_topk)
            correlations[layer_l] = float(corr) if not np.isnan(corr) else 1.0
        else:
            correlations[layer_l] = 1.0
    return correlations


# Backward-compatible alias
compute_adjacent_spearman_correlations = compute_spearman_correlations


def merge_average_spearman_correlations(
    per_sample_correlations: List[Dict[int, float]],
) -> Dict[int, float]:
    """Average per-layer ρ scalars across calibration samples (multi-sample L_det, option B)."""
    if not per_sample_correlations:
        return {}
    if len(per_sample_correlations) == 1:
        return dict(per_sample_correlations[0])
    all_layers = set()
    for corr in per_sample_correlations:
        all_layers.update(corr.keys())
    merged: Dict[int, float] = {}
    for layer in sorted(all_layers):
        vals = [c[layer] for c in per_sample_correlations if layer in c]
        if vals:
            merged[layer] = float(np.mean(vals))
    return merged


# Boundary for α_ℓ = ρ_ℓ - 2ρ_{ℓ-1} + ρ_{ℓ-2} when ℓ-1 or ℓ-2 < 0 (no measured pairs).
# Observation (Sec. 3): adjacent-layer Spearman ρ starts low, not high — use 0, not 1.
# (Distinct from layer-wise similarity s_ℓ, which is high in shallow layers.)
_RHO_PRIOR_BEFORE_LAYER0 = 0.0


def _rho_at_layer(correlations: Dict[int, float], layer: int) -> float:
    if layer in correlations:
        return correlations[layer]
    if layer < 0:
        return _RHO_PRIOR_BEFORE_LAYER0
    raise KeyError(f"Missing Spearman ρ for pair-start layer {layer}")


def compute_second_order_diffs(correlations: Dict[int, float]) -> Dict[int, float]:
    """
    α at pair-start ℓ = ρ_ℓ - 2ρ_{ℓ-1} + ρ_{ℓ-2} (Paper Sec. 4.2).

    For ℓ < 0, ρ is extrapolated as 0 (low pre-onset correlation). Enables α at ℓ=1
    when ρ_0, ρ_1 exist. L_det: first (+)→(−) on all ℓ ≤ L_end, then clamp to > L_start.
    """
    if not correlations:
        return {}
    max_l = max(correlations.keys())
    second_order_diffs: Dict[int, float] = {}
    for l in range(1, max_l + 1):
        if l not in correlations:
            continue
        second_order_diffs[l] = (
            _rho_at_layer(correlations, l)
            - 2.0 * _rho_at_layer(correlations, l - 1)
            + _rho_at_layer(correlations, l - 2)
        )
    return second_order_diffs


def select_detection_layer_from_correlations(
    correlations: Dict[int, float],
    L_start: int,
    L_end: int,
    num_layers: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[int, Dict[int, float]]:
    """
    Choose L_det from precomputed ρ (and derived α).

    1. α_ℓ = ρ_ℓ - 2ρ_{ℓ-1} + ρ_{ℓ-2}; ρ_{<0} := 0 so α exists from ℓ = 1 (plot only).
    2. ρ from pair L = 0; scan all pair-start ℓ with ℓ <= L_end for the first (+)→(−) α crossing.
    3. Paper L_det = ℓ*+1 is implemented as L_det = l_next at that crossing; then clamp:
       if L_det <= L_start → L_start + 1 (must detect strictly above stlayer).
    4. Else if first ℓ > L_start has α < 0 → L_start + 1; else min-α among ℓ > L_start.
    """
    second_order_diffs = compute_second_order_diffs(correlations)

    if verbose:
        corr_layers = sorted(correlations.keys())
        print(
            f"   Spearman ρ from pair L=0 (all {len(corr_layers)} pairs; "
            f"L_det search after stlayer L_start={L_start}):"
        )
        for l in corr_layers[:5]:
            print(f"     Pair L{l}-L{l+1}: ρ = {correlations[l]:.4f}")
        print(f"   Second-order α (sample, pair-start L <= L_end):")
        sample_layers = [
            l for l in sorted(second_order_diffs.keys()) if l <= L_end
        ][:8]
        for l in sample_layers:
            print(f"     Layer {l}: α = {second_order_diffs[l]:.4f}")

    if num_layers is None:
        max_pair_start = max(correlations.keys()) if correlations else L_end
        num_layers = max_pair_start + 2
    pair_end = min(L_end, num_layers - 2)
    scan_layers = sorted(l for l in second_order_diffs.keys() if l <= pair_end)
    post_start_layers = sorted(l for l in scan_layers if l > L_start)
    L_det = L_start + 1

    for i in range(len(scan_layers) - 1):
        l_curr = scan_layers[i]
        l_next = scan_layers[i + 1]
        if second_order_diffs[l_curr] >= 0 and second_order_diffs[l_next] < 0:
            L_det = l_next
            if verbose:
                print(
                    f"   Zero-crossing: α at pair-start {l_curr}→{l_next} "
                    f"→ L_det={L_det} (pre-clamp)"
                )
            break
    else:
        if post_start_layers:
            first_l = post_start_layers[0]
            if second_order_diffs[first_l] < 0:
                L_det = L_start + 1
                if verbose:
                    print(
                        f"   No zero-crossing; first pair-start > L_start ({first_l}) "
                        f"has α={second_order_diffs[first_l]:.4f} < 0 "
                        f"→ L_det = L_start+1 = {L_det}"
                    )
            else:
                min_alpha_layer = min(
                    post_start_layers, key=lambda l: second_order_diffs[l]
                )
                L_det = min_alpha_layer
                if verbose:
                    print(
                        f"   No zero-crossing; using min α at Layer {min_alpha_layer} "
                        f"(α={second_order_diffs[min_alpha_layer]:.4f})"
                    )

    if L_det <= L_start:
        if verbose:
            print(
                f"   Clamp: L_det={L_det} <= L_start={L_start} "
                f"→ L_det = L_start+1 = {L_start + 1}"
            )
        L_det = L_start + 1
    L_det = min(L_det, L_end)
    if verbose:
        print(f"   Detection layer L_det = {L_det}")

    return L_det, second_order_diffs


def auto_select_detection_layer(
    token_deviations: Optional[dict] = None,
    per_sample_token_deviations: Optional[List[dict]] = None,
    L_start: int = 0,
    L_end: int = 0,
    topk_ratio: float = LDET_TOPK_RATIO,
    verbose: bool = True,
    return_diagnostics: bool = False,
):
    """
    Automatically determine the Detection Layer L_det (Paper Section 4.2).

    Single sample: pass ``token_deviations`` only.
    Multi-sample (recommended): pass ``per_sample_token_deviations`` — compute ρ per sample,
    average ρ per layer, then α and L_det (not average deviation vectors first).

    Args:
        token_deviations: {layer_idx: deviation array} for one sample
        per_sample_token_deviations: list of per-sample deviation dicts
        L_start, L_end: stlayer and inclusive L_end; ρ from L=0; α scan ℓ <= L_end,
            then L_det = max(L_start + 1, …) if crossing falls at or below stlayer
        topk_ratio: top fraction of high-deviation tokens for Spearman
        verbose: print selection steps
        return_diagnostics: if True, return (L_det, correlations, second_order_diffs)

    Returns:
        L_det, or (L_det, correlations, second_order_diffs) when return_diagnostics=True
    """
    samples: List[dict] = []
    if per_sample_token_deviations:
        samples = list(per_sample_token_deviations)
    elif token_deviations is not None:
        samples = [token_deviations]

    if not samples:
        if verbose:
            print("  No token deviation data for L_det selection")
        out = L_start + 1
        return (out, {}, {}) if return_diagnostics else out

    if verbose:
        print(f"\n  Auto-selecting detection layer L_det...")
        print(
            f"   stlayer L_start={L_start}; α zero-crossing scan on pair-start "
            f"ℓ <= {L_end} (ρ from pair L=0); clamp L_det to > L_start if needed"
        )
        print(f"   TopK ratio: {topk_ratio*100:.0f}%")
        if len(samples) > 1:
            print(
                f"   Multi-sample: average per-layer ρ over {len(samples)} sample(s), "
                f"then compute α (option B)"
            )
        else:
            print("   Single-sample ρ → α")

    min_layers = min(len(sorted(s.keys())) for s in samples)
    if min_layers < 3:
        if verbose:
            print("  Need at least 3 layers to compute second-order difference")
        out = L_start + 1
        return (out, {}, {}) if return_diagnostics else out

    per_sample_rho = [
        compute_spearman_correlations(s, topk_ratio=topk_ratio) for s in samples
    ]
    correlations = merge_average_spearman_correlations(per_sample_rho)

    num_layers = max(max(s.keys()) for s in samples) + 1
    L_det, second_order_diffs = select_detection_layer_from_correlations(
        correlations, L_start, L_end, num_layers=num_layers, verbose=verbose,
    )

    if return_diagnostics:
        return L_det, correlations, second_order_diffs
    return L_det
