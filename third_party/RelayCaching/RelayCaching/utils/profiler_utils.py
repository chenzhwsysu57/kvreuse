import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, DynamicCache
)
import os, sys
# Get the absolute path of the directory containing this file
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# Add to Python search path so sibling modules (relay_utils.py, hijack files) can be found
sys.path.append(current_file_dir)
from fixed_mistral_hijack import enable_mistral_attention_prefill_context
from fixed_llama_hijack import enable_llama_attention_prefill_context
from fixed_qwen2_hijack import enable_qwen2_attention_prefill_context
from fixed_qwen3_hijack import enable_qwen3_attention_prefill_context
import pickle
import io
import numpy as np
import random
from langchain.prompts import PromptTemplate
import time
MAP_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Analyze the context step-by-step to find information related to: "{question}".
    
    First, identify all segments of the text that touch upon the topic. 
    Second, expand on these segments by including the necessary details, definitions, and any related data mentioned. 
    Ensure the output is a verbose and thorough summary of the relevant parts.

    **Context:**
    {context}

    **Thorough Extraction:**
    """
)

FILTER_PHRASES = [
    "not contain information",
    "not contain any information",
    "not provide information",
    "no information regarding",
    "information not available",
    "information not found",
    "not contain relevant information",
    "no relevant information",
    "information not have",
    "is not mentioned",
    "not include any details",
    "not provide specific information",
    "not contain relevent information",
]

# Reduce Prompt Templates for Layer Profiler
REDUCE_PROMPT_PREFIX_TEMPLATE = """
Answer the question based on the given informations. Only give me the answer and do not output any other words.
Question:{question}
The following are given infos:

"""

REDUCE_PROMPT_SUFFIX_TEMPLATE = """
You are an expert information extraction assistant. Your task is to analyze the provided long-context materials meticulously. 
Strictly adhere to the following constraints:
- Use only the information provided in the preceding context; do not use external knowledge.
- The output must be extremely laconic and not exceed a five-word limit.
- Punctuation should be minimized.
- Do not use complete sentences if a phrase suffices.
- Avoid any introductory phrases like "Based on the text" or "The answer is".

Question: {question}
Strictly formatted Answer:"""

def flatten_list(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            # Recursively expand sub-lists
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def compute_accumulate_atten(head_attn):
    # Convert to higher precision for computation
    return head_attn.sum(dim=1).to(head_attn.dtype)


def compute_accumulate_attention(matrix, pattern):
    # (layer, head, token, token)
    layer_res = []
    for layer_atten in matrix:
        head_res = []
        for head_attn in layer_atten:
            if pattern == "value":
                temp_res = compute_accumulate_atten(head_attn)
            elif pattern == "key":
                temp_res = compute_accumulate_atten(head_attn.transpose(0, 1))
            head_res.append(temp_res)
        layer_res.append(torch.stack(head_res))
    return torch.stack(layer_res)

def extract_generated_self_atten(attentions, prefix_len, suffix_len, batch_idx = 0):
    """
    Extract self-attention within the generated (suffix) portion.
    """
    layer_attns = []
    for layer_attn in attentions:
        # Slice: both Query and Key belong to the generated portion (suffix)
        self_attn = layer_attn[batch_idx, :, prefix_len:prefix_len+suffix_len, prefix_len:prefix_len+suffix_len]
        layer_attns.append(self_attn)
    return layer_attns

def extract_prefix_atten_with_tokens(attentions, prefix_len, suffix_len, batch_idx = 0):
    layer_attns = []
    for layer_attn in attentions:
        suffix_to_prefix_attn = layer_attn[batch_idx, :, prefix_len:prefix_len+suffix_len, :prefix_len]

        # Apply mask and normalize
        masked_attn = suffix_to_prefix_attn
        epsilon = 1e-6
        masked_attn = masked_attn / (masked_attn.sum(dim=-2, keepdim=True) + epsilon)
        layer_attns.append(masked_attn)
    return layer_attns

def compute_recompute_indices_combine(
    attentions: Tuple[torch.Tensor],
    inputs_len: int,
    generated_len: int,
    pre_window_size: int = 5,
    suf_window_size: int = 5,
    prefix_influence_ratio: float = 0.1,
    downstream_influence_ratio: float = 0.1,
) -> List[int]:
    """
    Compute recompute indices based on four criteria:
    1. Pre-window (count-specified)
    2. Post-window (count-specified)
    3. Positions heavily influenced by prefix (ratio-specified)
    4. Positions with high downstream influence (ratio-specified)
    """
    actual_pre_window = min(pre_window_size, generated_len)
    pre_window_indices = list(range(actual_pre_window))

    actual_suf_window = min(suf_window_size, generated_len)
    suf_window_indices = list(range(max(0, generated_len - actual_suf_window), generated_len))
    # Extract "generated -> prefix" attention matrix
    attn_to_prefix = extract_prefix_atten_with_tokens(attentions, inputs_len, generated_len)
    
    # Compute "receiving" score for each generated token
    # Output shape: [layer, head, generated_len]
    prefix_influence_scores = compute_accumulate_attention(attn_to_prefix, "value")
    kernel_size = 5
    # Aggregate scores across all layers and heads
    aggregated_prefix_scores = torch.sum(prefix_influence_scores, dim=(0, 1)).cpu()
    # Apply smoothing
    if kernel_size > 0 and generated_len > kernel_size:
        prefix_influence_scores = prefix_influence_scores.reshape(1, 1, -1)
        # Use padding to preserve length after smoothing
        prefix_influence_scores = F.avg_pool1d(prefix_influence_scores, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
        prefix_influence_scores = prefix_influence_scores.reshape(-1)
    k_prefix_div2 = int(generated_len * prefix_influence_ratio / 2)
    prefix_top_indices = torch.argsort(aggregated_prefix_scores, descending=True)[:k_prefix_div2].tolist()
    prefix_low_indices = torch.argsort(aggregated_prefix_scores)[:k_prefix_div2].tolist()
    prefix_indices = prefix_top_indices + prefix_low_indices

    attn_within_generated = extract_generated_self_atten(attentions, inputs_len, generated_len)
    
    # Compute "emitting" score for each generated token
    # Output shape: [layer, head, generated_len]
    downstream_influence_scores = compute_accumulate_attention(attn_within_generated, "key")
    # Aggregate scores across all layers and heads
    aggregated_downstream_scores = torch.sum(downstream_influence_scores, dim=(0, 1)).cpu()
    # Apply pooling for smoothing
    if kernel_size > 0 and generated_len > kernel_size:
        downstream_influence_scores = downstream_influence_scores.reshape(1, 1, -1)
        # Use padding to preserve length after smoothing
        downstream_influence_scores = F.avg_pool1d(downstream_influence_scores, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
    downstream_influence_scores = downstream_influence_scores.reshape(-1)
    k_downstream = int(generated_len * downstream_influence_ratio)
    downstream_indices = torch.argsort(aggregated_downstream_scores, descending=True)[:k_downstream].tolist()
    final_indices = downstream_indices
    return final_indices


def map_model_name(model_name):
        if "llama" in model_name.lower():
            return "llama"
        elif "mistral" in model_name.lower():
            return "mistral"
        elif "gemma" in model_name.lower():
            return "gemma"
        elif "qwen2" in model_name.lower():
            return "qwen2"
        elif "qwen3" in model_name.lower():
            return "qwen3"
        
def debug_allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-5, name="Tensor"):
    """
    Check if two tensors are close, and print mismatched positions sorted by difference magnitude.
    """
    close_mask = torch.isclose(tensor_a, tensor_b, atol=atol, rtol=rtol)
    mismatched_mask = ~close_mask

    if not mismatched_mask.any():
        print(f"  {name} tensors are close.")
        return True

    mismatched_indices = torch.nonzero(mismatched_mask)
    
    abs_diff = torch.abs(tensor_a - tensor_b)
    mismatched_diffs = abs_diff[mismatched_mask]

    sorted_diffs, sorted_indices_of_mismatches = torch.sort(mismatched_diffs, descending=True)
    
    sorted_mismatched_indices = mismatched_indices[sorted_indices_of_mismatches]

    print(f"  Mismatches found in {name}!")
    print(f"Total mismatched elements: {mismatched_indices.shape[0]}")
    print("Showing top 10 largest differences:")

    for i in range(min(10, len(sorted_diffs))):
        # Get current index and difference value
        index = sorted_mismatched_indices[i]
        diff_val = sorted_diffs[i]
        
        # Convert index to tuple for tensor indexing
        idx_tuple = tuple(index.tolist())
        val_a = tensor_a[idx_tuple]
        val_b = tensor_b[idx_tuple]
        
        print(f"  - Rank {i+1} | Index {str(idx_tuple):<25} | A={val_a:<10.6f} | B={val_b:<10.6f} | Diff={diff_val:<10.6f}")
    
    return False

def swap_dict_values(data_map: dict, key1, key2) -> bool:
    """
    Swaps the values of two keys in a dictionary in-place.

    Args:
        data_map (dict): The dictionary to modify.
        key1: The first key.
        key2: The second key.

    Returns:
        bool: True if the swap was successful, False otherwise.
    """
    if key1 not in data_map:
        print(f"Error: Key '{key1}' not found in the dictionary.")
        return False
    if key2 not in data_map:
        print(f"Error: Key '{key2}' not found in the dictionary.")
        return False

    if key1 == key2:
        print("Warning: Both keys are the same. No swap performed.")
        return True

    print(f"Swapping values for keys '{key1}' and '{key2}'...")
    data_map[key1], data_map[key2] = data_map[key2], data_map[key1]
    
    return True

def swap_list_elements(data_list: list, index1: int, index2: int) -> bool:
    """
    Swaps the elements at two given indices in a list in-place.
    Supports both positive and negative indices.

    Args:
        data_list (list): The list to modify.
        index1 (int): The first index.
        index2 (int): The second index.

    Returns:
        bool: True if the swap was successful, False otherwise.
    """
    list_len = len(data_list)

    if not (-list_len <= index1 < list_len):
        print(f"Error: index1 ({index1}) is out of bounds for a list of size {list_len}.")
        return False
    
    if not (-list_len <= index2 < list_len):
        print(f"Error: index2 ({index2}) is out of bounds for a list of size {list_len}.")
        return False

    if index1 == index2:
        print("Warning: Both indices are the same. No swap performed.")
        return True

    print(f"Swapping elements at index {index1} and {index2}...")
    data_list[index1], data_list[index2] = data_list[index2], data_list[index1]
    
    return True

def flatten_list(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            # Recursively expand sub-lists
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def get_device():
    """Automatically detect available compute device (priority: CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def enable_model_attention_prefill_context(model_name: str):
        if "mistral" in model_name.lower():
            enable_mistral_attention_prefill_context(model_name)
        elif "llama" in model_name.lower():
            enable_llama_attention_prefill_context(model_name)
        elif "qwen2" in model_name.lower():
            enable_qwen2_attention_prefill_context(model_name)
        elif "r1-distill-qwen-32b" in model_name.lower():
            enable_qwen2_attention_prefill_context(model_name)
        elif "qwen3" in model_name.lower() and "a3b" in model_name.lower():
            raise ValueError(
                "Qwen3 MoE checkpoints (e.g. Qwen3-30B-A3B) are not supported in this release. "
                "Use a dense model: Llama-3.1, Mistral-7B, Qwen2.5, or dense Qwen3."
            )
        elif "qwen3" in model_name.lower():
            enable_qwen3_attention_prefill_context(model_name)
def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)

def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"
    return question[0].lower() + question[1:]


def build_prefix(model_name, prompt):
    model_name = model_name.lower()  # normalize for stability
    if "deepseek" in model_name:
        prompt = f"<｜begin of sentence｜><｜User｜>{prompt}"
    elif "llama" in model_name:
        prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}"
    elif "mistral" in model_name:
        prompt = f"<s>[INST]{prompt}"
    elif "gemma" in model_name:
        prompt = f"<bos><start_of_turn>user\n{prompt}"
    elif "qwen" in model_name:
        prompt = f"<|im_start|>user\n{prompt}"
    return prompt

def build_suffix(model_name, prompt):
    model_name = model_name.lower()  # normalize for stability
    if "deepseek" in model_name:
        prompt = f"{prompt}<｜Assistant｜><think>\n"
    elif "llama" in model_name:
        prompt = f"{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    elif "mistral" in model_name:
        prompt = f"{prompt}[/INST]"
    elif "gemma" in model_name:
        prompt = f"{prompt}<end_of_turn>\n<start_of_turn>model\n"   
    elif "qwen" in model_name:
        prompt = f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prompt

class ROPE_Module:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

        # Load model configuration
        self.config = AutoConfig.from_pretrained(model_name)
        # Determine model type
        self.model_type = self._determine_model_type()
        # Load model-specific RoPE implementation
        self._load_model_specific_components()

        # Save original max position embeddings
        self.max_position_embeddings = self.config.max_position_embeddings
        # Initialize rotary embedding layer
        self.rotary_emb = self._init_rotary_embedding()

    def _determine_model_type(self):
        """Determine model type (e.g., llama, qwen, etc.)"""
        if (
            "llama" in self.model_name.lower()
            or "llama" in self.config.architectures[0].lower()
        ):
            return "llama"
        elif (
            "qwen2" in self.model_name.lower()
            or "qwen2" in self.config.architectures[0].lower()
        ):
            return "qwen2"
        elif (
            "mistral" in self.model_name.lower()
            or "mistral" in self.config.architectures[0].lower()
        ):
            return "mistral"
        elif (
            "qwen3" in self.model_name.lower()
            or "qwen3" in self.config.architectures[0].lower()
        ):
            return "qwen3"
        else:
            raise ValueError(f"Unsupported model type: { self.model_name}")


    def _load_model_specific_components(self):
        """Load model-specific components and functions"""
        if self.model_type == "llama":
            from transformers.models.llama.modeling_llama import (
                LlamaRotaryEmbedding, apply_rotary_pos_emb)

            self.RotaryEmbeddingClass = LlamaRotaryEmbedding
            self.apply_rotary_pos_emb = apply_rotary_pos_emb

        elif self.model_type == "qwen2":
            from transformers.models.qwen2.modeling_qwen2 import (
                Qwen2RotaryEmbedding, apply_rotary_pos_emb)

            self.RotaryEmbeddingClass = Qwen2RotaryEmbedding
            self.apply_rotary_pos_emb = apply_rotary_pos_emb

        elif self.model_type == "mistral":
            from transformers.models.mistral.modeling_mistral import (
                MistralRotaryEmbedding, apply_rotary_pos_emb)

            self.RotaryEmbeddingClass = MistralRotaryEmbedding
            self.apply_rotary_pos_emb = apply_rotary_pos_emb
        elif self.model_type == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import (
                Qwen3RotaryEmbedding, apply_rotary_pos_emb)

            self.RotaryEmbeddingClass = Qwen3RotaryEmbedding
            self.apply_rotary_pos_emb = apply_rotary_pos_emb
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _init_rotary_embedding(self):
        """Initialize rotary position embedding layer"""
        # head_dim = self.config.hidden_size // self.config.num_attention_heads
        return self.RotaryEmbeddingClass(self.config).to(self.device)

    def apply_rope(self, x, position_ids):
        """
        Apply rotary position embedding.

        Args:
            x: Key tensor
            position_ids: Position IDs
        """

        cos, sin = self.rotary_emb(x, position_ids)
        query = torch.zeros_like(x)
        # Apply RoPE to query and key respectively
        _, k = self.apply_rotary_pos_emb(query, x, cos, sin)
        return k

    def inverse_rope(self, x, position_ids):
        """Inverse RoPE: recover rotated vectors to their original state."""
        cos, sin = self.rotary_emb(x, position_ids)
        query = torch.zeros_like(x)
        # Apply inverse rotation (note the sign reversal)
        _, k = self.apply_rotary_pos_emb(query, x, cos, -sin)
        return k
    
def load_model(
    model_name: str,
    quantization: Optional[str] = None,
    device_map: Optional[str] = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Unified LLM model loading function with support for different quantization levels and device mapping.

    Args:
        model_name: Model name or path
        quantization: Quantization level, options: None, "8bit", "4bit"
        device_map: Device mapping strategy, e.g., "auto", "cuda:0", "mps", "cpu"
        torch_dtype: Model data type, e.g., torch.float16, torch.bfloat16
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        model: Loaded model
        tokenizer: Corresponding tokenizer
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    # Handle quantization config
    quantization_config = None
    if quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.float16,
        )
    elif quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Set default dtype if not specified
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **kwargs,
    ).eval()
    return model, tokenizer

def extract_KV_from_outputs(
    output_kv_cache: DynamicCache, prefix_len: int, context_len: int
):
    new_kv_cache = []
    for layer in output_kv_cache:
        k, v = layer
        k_slice = k[:, :, prefix_len : prefix_len + context_len, :].clone()
        v_slice = v[:, :, prefix_len : prefix_len + context_len, :].clone()
        new_kv_cache.append([k_slice, v_slice])
        del k, v
    return DynamicCache(new_kv_cache)

def extract_kv_with_tokens(outputs, prefix_len, suffix_len, batch_idx = 0):
    layer_kvs = []
    for layer_kv in outputs.past_key_values:
        key = layer_kv[0][batch_idx]
        value = layer_kv[1][batch_idx]
        key_suffix = key[:, prefix_len : prefix_len + suffix_len, :].detach().cpu()
        value_suffix = value[:, prefix_len : prefix_len + suffix_len, :].detach().cpu()
        layer_kvs.append({"value": value_suffix, "key": key_suffix})
    return layer_kvs

def kv_cache_apply_rope(
    kv_cache: DynamicCache, 
    position_ids: torch.Tensor, 
    rope_module: ROPE_Module
):
    for layer_key in kv_cache.key_cache:
        batch_size, head_size, _, _ = layer_key.shape
        for b in range(batch_size):
            for h in range(head_size):
                key_vector = layer_key[b, h]
                key_vector_added = rope_module.apply_rope(
                    key_vector, position_ids=position_ids
                )
                layer_key[b, h] = key_vector_added
    return kv_cache

def kv_cache_inverse_rope(
    kv_cache: DynamicCache, 
    position_ids: torch.Tensor, 
    rope_module: ROPE_Module
):
    for layer_key in kv_cache.key_cache:
        batch_size, head_size, _, _ = layer_key.shape
        for b in range(batch_size):
            for h in range(head_size):
                key_vector = layer_key[b, h].detach().clone()
                key_vector_removed = rope_module.inverse_rope(
                    key_vector, position_ids=position_ids
                )
                layer_key[b, h] = key_vector_removed
    return kv_cache

def calculate_overall_similarity(model1_all_layers, model2_all_layers, remove_first=0):
    """
    Compute overall KV similarity between two per-layer cache lists.

    Per head: token-wise cosine for K/V; norm ratio = ||model2|| / (||model1|| + eps).
    Per layer: mean over heads; norm similarity = 1 - mean(|ratio - 1|).
    Returns layer-averaged metrics (callers pass reference as model1, candidate as model2).

    Args:
        model1_all_layers: Baseline KV per layer (e.g. gold / prefill reference).
        model2_all_layers: Candidate KV per layer (e.g. decoding / repaired).
        remove_first: Tokens to drop from the start of each head sequence.

    Returns:
        Dict with key_cosine_similarity, value_cosine_similarity,
        key_norm_similarity, value_norm_similarity; empty dict on error.
    """
    num_layers = len(model1_all_layers)
    if num_layers != len(model2_all_layers):
        print("Error: Layer count mismatch between two models.")
        return {}
    if num_layers == 0:
        print("Warning: Model layer list is empty.")
        return {}

    all_layers_metrics = []

    for layer_idx in range(num_layers):
        
        layer_kv1 = model1_all_layers[layer_idx]
        layer_kv2 = model2_all_layers[layer_idx]
        
        if "key" not in layer_kv1 or "key" not in layer_kv2:
            print(f"Layer {layer_idx}: Error: 'key' missing from input dict. Skipping.")
            all_layers_metrics.append(None)
            continue
            
        num_heads = len(layer_kv1["key"])
        if num_heads != len(layer_kv2["key"]):
            print(f"Layer {layer_idx}: Error: Head count mismatch. Skipping.")
            all_layers_metrics.append(None)
            continue

        all_head_key_sims = []
        all_head_value_sims = []
        all_key_norm_ratio = []
        all_value_norm_ratio = []

        for head_idx in range(num_heads):
            k1 = layer_kv1["key"][head_idx][remove_first:]
            v1 = layer_kv1["value"][head_idx][remove_first:]
            k2 = layer_kv2["key"][head_idx][remove_first:]
            v2 = layer_kv2["value"][head_idx][remove_first:]

            if k1.shape[0] == 0 or k1.shape != k2.shape or k1.dim() != 2:
                continue

            token_wise_key_cos = F.cosine_similarity(k1, k2, dim=1)
            token_wise_value_cos = F.cosine_similarity(v1, v2, dim=1)
            avg_key_cos_for_this_head = token_wise_key_cos.mean().item()
            avg_value_cos_for_this_head = token_wise_value_cos.mean().item()
            
            epsilon = 1e-8
            k1_norms = torch.norm(k1, p=2, dim=1)
            k2_norms = torch.norm(k2, p=2, dim=1)
            v1_norms = torch.norm(v1, p=2, dim=1)
            v2_norms = torch.norm(v2, p=2, dim=1)
            # Per-token amplitude ratio (model2 vs model1), then |ratio-1| for norm similarity.
            token_wise_key_norm_ratio = k2_norms / (k1_norms + epsilon)
            token_wise_value_norm_ratio = v2_norms / (v1_norms + epsilon)
            avg_key_norm_ratio_for_this_head = token_wise_key_norm_ratio.mean().item()
            avg_value_norm_ratio_for_this_head = token_wise_value_norm_ratio.mean().item()

            all_head_key_sims.append(avg_key_cos_for_this_head)
            all_head_value_sims.append(avg_value_cos_for_this_head)
            all_key_norm_ratio.append(avg_key_norm_ratio_for_this_head)
            all_value_norm_ratio.append(avg_value_norm_ratio_for_this_head)
        
        if not all_head_key_sims:
            print(f"Layer {layer_idx}: Warning: no comparable heads. Skipping this layer.")
            all_layers_metrics.append(None)
            continue
            
        layer_key_cos = np.mean(all_head_key_sims)
        layer_value_cos = np.mean(all_head_value_sims)
        layer_key_norm_sim = 1 - np.mean(np.abs(np.array(all_key_norm_ratio) - 1))
        layer_value_norm_sim = 1 - np.mean(np.abs(np.array(all_value_norm_ratio) - 1))
        
        layer_metrics = {
            "key_cosine_similarity": layer_key_cos,
            "value_cosine_similarity": layer_value_cos,
            "key_norm_similarity": layer_key_norm_sim,
            "value_norm_similarity": layer_value_norm_sim
        }
        
        all_layers_metrics.append(layer_metrics)
        
    valid_layer_metrics = [m for m in all_layers_metrics if m is not None]
    if not valid_layer_metrics:
        print("Warning: no comparable data in any layer.")
        return {}
    
    final_avg_key_cos = np.mean([m['key_cosine_similarity'] for m in valid_layer_metrics])
    final_avg_value_cos = np.mean([m['value_cosine_similarity'] for m in valid_layer_metrics])
    final_avg_key_norm = np.mean([m['key_norm_similarity'] for m in valid_layer_metrics])
    final_avg_value_norm = np.mean([m['value_norm_similarity'] for m in valid_layer_metrics])

    final_average_metrics = {
        "key_cosine_similarity": final_avg_key_cos,
        "value_cosine_similarity": final_avg_value_cos,
        "key_norm_similarity": final_avg_key_norm,
        "value_norm_similarity": final_avg_value_norm
    }
    return final_average_metrics

class CPU_Unpickler(pickle.Unpickler):
    """Custom unpickler to load tensors on CPU."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
