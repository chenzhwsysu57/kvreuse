"""
Observation Data Collector
===========================

Collects and stores KV cache similarity metrics during observation experiments.
Provides methods to compute layer-wise, token-wise, and macro-level similarities.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Any

_EPS = 1e-8


def _squeeze_heads(k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize KV tensors to shape [H, N, D]."""
    if k.dim() == 4:
        k, v = k.squeeze(0), v.squeeze(0)
    return k, v


def _align_kv_seq_len(
    k_a: torch.Tensor,
    v_a: torch.Tensor,
    k_b: torch.Tensor,
    v_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncate to common sequence length (dim -2) before similarity computation."""
    if k_a.dim() == 3:
        n = min(k_a.shape[1], k_b.shape[1])
        return k_a[:, :n], v_a[:, :n], k_b[:, :n], v_b[:, :n]
    n = min(k_a.shape[-2], k_b.shape[-2])
    return k_a[..., :n, :], v_a[..., :n, :], k_b[..., :n, :], v_b[..., :n, :]


def _kv_tuples_to_layer_dicts(
    kv_list: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Dict[str, List[torch.Tensor]]]:
    """Convert per-layer (key, value) tuples to calculate_overall_similarity format."""
    layers = []
    for k, v in kv_list:
        k, v = _squeeze_heads(k, v)
        layers.append({
            'key': [k[head_idx] for head_idx in range(k.shape[0])],
            'value': [v[head_idx] for head_idx in range(v.shape[0])],
        })
    return layers


def compute_macro_pair(
    kv_a: List[Tuple[torch.Tensor, torch.Tensor]],
    kv_b: List[Tuple[torch.Tensor, torch.Tensor]],
) -> dict:
    """
    Macro similarity for one pair (model1=kv_a, model2=kv_b).

    Uses profiler_utils.calculate_overall_similarity — same as original Step 8.
    """
    from RelayCaching.utils.profiler_utils import calculate_overall_similarity

    aligned_a = []
    aligned_b = []
    for (ka, va), (kb, vb) in zip(kv_a, kv_b):
        ka, va, kb, vb = _align_kv_seq_len(ka, va, kb, vb)
        aligned_a.append((ka, va))
        aligned_b.append((kb, vb))

    overall = calculate_overall_similarity(
        _kv_tuples_to_layer_dicts(aligned_a),
        _kv_tuples_to_layer_dicts(aligned_b),
    )
    if not overall:
        return {
            'key_cosine': 0.0,
            'value_cosine': 0.0,
            'key_norm': 0.0,
            'value_norm': 0.0,
        }
    return _overall_to_collector_metrics(overall)


def _overall_to_collector_metrics(overall: Dict[str, float]) -> dict:
    return {
        'key_cosine': overall['key_cosine_similarity'],
        'value_cosine': overall['value_cosine_similarity'],
        'key_norm': overall['key_norm_similarity'],
        'value_norm': overall['value_norm_similarity'],
    }


def _layer_similarity_metrics(
    k_a: torch.Tensor,
    v_a: torch.Tensor,
    k_b: torch.Tensor,
    v_b: torch.Tensor,
) -> dict:
    """
    Per-layer metrics matching profiler_utils.calculate_overall_similarity.

    model1 = (k_a, v_a), model2 = (k_b, v_b); ratio = ||model2|| / ||model1|| per token,
    averaged within each head, then aggregated across heads.
    """
    k_a, v_a = _squeeze_heads(k_a, v_a)
    k_b, v_b = _squeeze_heads(k_b, v_b)

    all_head_key_sims = []
    all_head_value_sims = []
    all_key_norm_ratio = []
    all_value_norm_ratio = []

    for head_idx in range(k_a.shape[0]):
        k1 = k_a[head_idx]
        k2 = k_b[head_idx]
        v1 = v_a[head_idx]
        v2 = v_b[head_idx]

        if k1.shape[0] == 0 or k1.shape != k2.shape or k1.dim() != 2:
            continue

        token_wise_key_cos = F.cosine_similarity(k1, k2, dim=1)
        token_wise_value_cos = F.cosine_similarity(v1, v2, dim=1)
        all_head_key_sims.append(token_wise_key_cos.mean().item())
        all_head_value_sims.append(token_wise_value_cos.mean().item())

        k1_norms = torch.norm(k1, p=2, dim=1)
        k2_norms = torch.norm(k2, p=2, dim=1)
        v1_norms = torch.norm(v1, p=2, dim=1)
        v2_norms = torch.norm(v2, p=2, dim=1)

        all_key_norm_ratio.append((k2_norms / (k1_norms + _EPS)).mean().item())
        all_value_norm_ratio.append((v2_norms / (v1_norms + _EPS)).mean().item())

    if not all_head_key_sims:
        return {
            'key_cosine': 0.0,
            'value_cosine': 0.0,
            'key_norm': 0.0,
            'value_norm': 0.0,
        }

    return {
        'key_cosine': float(np.mean(all_head_key_sims)),
        'value_cosine': float(np.mean(all_head_value_sims)),
        'key_norm': float(1.0 - np.mean(np.abs(np.array(all_key_norm_ratio) - 1.0))),
        'value_norm': float(1.0 - np.mean(np.abs(np.array(all_value_norm_ratio) - 1.0))),
    }


class ObservationDataCollector:
    """
    Collects similarity data during observation experiments.

    Supports three granularities:
    - macro: overall average
    - layer-wise: per-layer average similarity
    - token-wise: per-token similarity

    Args:
        num_layers: number of model layers
        save_dir: directory to save collected data
    """
    
    def __init__(self, num_layers: int, save_dir: str = "./observation_data"):
        self.num_layers = num_layers
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.layer_similarities = []
        self.token_similarities = {}
        self.macro_similarities = {}

    def collect_layer_wise_similarity(
        self,
        decoding_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        prefill_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """
        Collect layer-wise similarity metrics.

        Args:
            decoding_kv: list of (key, value) tuples per layer, shape [1, H, N, D]
            prefill_kv: list of (key, value) tuples per layer, shape [1, H, N, D]
        """
        self.layer_similarities = []
        
        for layer_idx in range(self.num_layers):
            dec_k, dec_v = decoding_kv[layer_idx]
            pre_k, pre_v = prefill_kv[layer_idx]
            dec_k, dec_v, pre_k, pre_v = _align_kv_seq_len(dec_k, dec_v, pre_k, pre_v)
            self.layer_similarities.append(
                _layer_similarity_metrics(dec_k, dec_v, pre_k, pre_v)
            )
    
    def collect_token_wise_similarity(
        self,
        decoding_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        prefill_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        """
        Collect token-level similarity metrics.

        Args:
            decoding_kv: list of (key, value) tuples per layer, shape [1, H, N, D]
            prefill_kv: list of (key, value) tuples per layer, shape [1, H, N, D]
        """
        self.token_similarities = {}
        
        for layer_idx in range(self.num_layers):
            dec_k, dec_v = decoding_kv[layer_idx]
            pre_k, pre_v = prefill_kv[layer_idx]
            dec_k, dec_v, pre_k, pre_v = _align_kv_seq_len(dec_k, dec_v, pre_k, pre_v)
            dec_k, dec_v = _squeeze_heads(dec_k, dec_v)
            pre_k, pre_v = _squeeze_heads(pre_k, pre_v)
            
            key_cos = F.cosine_similarity(dec_k, pre_k, dim=-1).mean(dim=0)
            value_cos = F.cosine_similarity(dec_v, pre_v, dim=-1).mean(dim=0)
            
            dec_k_norm = torch.norm(dec_k, p=2, dim=-1)
            pre_k_norm = torch.norm(pre_k, p=2, dim=-1)
            dec_v_norm = torch.norm(dec_v, p=2, dim=-1)
            pre_v_norm = torch.norm(pre_v, p=2, dim=-1)

            key_ratio = pre_k_norm / (dec_k_norm + _EPS)
            value_ratio = pre_v_norm / (dec_v_norm + _EPS)
            key_norm = (1.0 - torch.abs(key_ratio - 1.0)).mean(dim=0)
            value_norm = (1.0 - torch.abs(value_ratio - 1.0)).mean(dim=0)
            
            self.token_similarities[layer_idx] = {
                'key_cosine': key_cos.float().cpu().numpy(),
                'value_cosine': value_cos.float().cpu().numpy(),
                'key_norm': key_norm.float().cpu().numpy(),
                'value_norm': value_norm.float().cpu().numpy(),
            }
    
    def collect_macro_similarity(
        self,
        decoding_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        prefill_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        random_kv: List[Tuple[torch.Tensor, torch.Tensor]] = None,
        independent_kv: List[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Collect macro-level similarity data across multiple KV sources.

        Fig. 1 compares each KV source against Prefill KV (ground truth):
        - decoding_vs_prefill
        - random_vs_prefill
        - independent_vs_prefill

        decoding_vs_random is kept for diagnostics only (not used in Fig. 1).

        Args:
            decoding_kv: Decoding KV (our method)
            prefill_kv: Full-prefill KV (ground truth)
            random_kv: Random KV baseline
            independent_kv: Independent prefill KV baseline
        """
        self.macro_similarities = {}
        
        self.macro_similarities['decoding_vs_prefill'] = compute_macro_pair(decoding_kv, prefill_kv)
        
        if random_kv is not None:
            # Fig. 1 Random column: Random KV vs Prefill KV (original observation_data JSON schema)
            self.macro_similarities['random_vs_prefill'] = compute_macro_pair(random_kv, prefill_kv)
            # Diagnostic only (original also logged decoding_vs_random)
            self.macro_similarities['decoding_vs_random'] = compute_macro_pair(decoding_kv, random_kv)
        
        if independent_kv is not None:
            self.macro_similarities['independent_vs_prefill'] = compute_macro_pair(independent_kv, prefill_kv)
    
    def save_data(self, filename: str = "observation_data.json"):
        """
        Save collected data to a JSON file.

        Args:
            filename: output filename
        """
        data = {
            'layer_similarities': self.layer_similarities,
            'token_similarities': {
                str(k): {
                    metric: v[metric].tolist() if isinstance(v[metric], np.ndarray) else v[metric]
                    for metric in v
                }
                for k, v in self.token_similarities.items()
            },
            'macro_similarities': self.macro_similarities,
        }
        
        save_path = self.save_dir / filename
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"   Observation data saved to: {save_path}")
        return str(save_path)
    
    @classmethod
    def load_data(cls, data_path: str) -> dict:
        """
        Load data from a JSON file.

        Args:
            data_path: path to the data file

        Returns:
            dict: loaded data
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'token_similarities' in data:
            for layer_key, metrics in data['token_similarities'].items():
                for metric_name, values in metrics.items():
                    if isinstance(values, list):
                        data['token_similarities'][layer_key][metric_name] = np.array(values)
        
        return data
