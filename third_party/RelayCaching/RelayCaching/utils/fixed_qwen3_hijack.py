import inspect
import types
import warnings
from typing import List, Optional, Tuple, Union
import os, sys
# Resolve project paths for fallback imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(project_root, 'utils')

if utils_dir not in sys.path:
    sys.path.append(utils_dir)
import torch
import transformers
try:
    from RelayCaching.utils.relay_utils import RelayCache
except ImportError:
    # Fallback to relative import if absolute import fails
    from relay_utils import RelayCache
from transformers.cache_utils import Cache, DynamicCache, StaticCache, SlidingWindowCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
)

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.processing_utils import Unpack
from collections.abc import Callable
from transformers.utils import logging
logger = logging.get_logger(__name__)

def enable_qwen3_attention_prefill_context(model_name):
    """
    Hijack Qwen3 model to use RelayCache for KV cache reuse.
    """
    assert "qwen" in model_name.lower(), "Model is not Qwen3"
    transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen3
    transformers.models.qwen3.modeling_qwen3.Qwen3Model.forward = fixed_Qwen3Model_forward
    transformers.models.qwen3.modeling_qwen3.Qwen3Model._prepare_4d_causal_attention_mask_with_cache_position = new_prepare_4d_causal_attention_mask_with_cache_position
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = qwen3_attention_forward
    transformers.models.qwen3.modeling_qwen3.Qwen3Model._update_causal_mask = new_update_causal_mask


def qwen3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if isinstance(past_key_value, RelayCache):
            if past_key_value.is_decoding(query_states):
                key_states, value_states = past_key_value.update_decode(
                    query_states, key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_states, value_states = past_key_value.update_prefill(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    # Default SDPA: same backend as model load and all compared methods (README Lesson 2).
    # SDPA/Flash-style kernels do not expose attention scores; eager does, but switching backends
    # changes numerics. For token-level influence (tau_inf), recompute scores under SDPA instead
    # of replacing this with eager_attention_forward.
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
            "Continuing with SDPA, but output_attentions will be ignored."
        )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def fixed_Qwen3Model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None, 
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> BaseModelOutputWithPast:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError(
            "The `past_key_values` should be either a `Cache` object or `None`."
        )

    _relay_skip_embed = False
    _relay_full_input_ids = None
    if (
        inputs_embeds is None
        and isinstance(past_key_values, RelayCache)
        and getattr(past_key_values, 'execution_mode', None) == 'relaycaching'
        and getattr(past_key_values, 'recompute_label', False)
        and getattr(past_key_values, 'reuse_hidden_layer', -1) >= 0
        and getattr(past_key_values, 'reusable_hidden_states', None) is not None
        and input_ids is not None
        and input_ids.shape[1] > 1
    ):
        relay_rel = past_key_values.relay_relative_indices_tensor
        if relay_rel is not None and relay_rel.numel() > 0:
            _relay_skip_embed = True
            _relay_full_input_ids = input_ids
            full_seq_len = input_ids.shape[1]
            keep_indices = past_key_values._get_keep_indices_fast(relay_rel, full_seq_len)
            inputs_embeds = self.embed_tokens(input_ids[:, keep_indices])
            past_key_values._skip_initial_drop = True

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if _relay_skip_embed and _relay_full_input_ids is not None:
        _full_seq_len = _relay_full_input_ids.shape[1]
        _dummy_input_tensor = inputs_embeds.new_empty(
            inputs_embeds.shape[0], _full_seq_len, inputs_embeds.shape[2]
        )
        causal_mask = self._update_causal_mask(
            attention_mask,
            _dummy_input_tensor,
            cache_position,
            past_key_values,
            output_attentions,
        )
        position_embeddings = self.rotary_emb(_dummy_input_tensor, position_ids)
    else:
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    
    is_relay_cache = isinstance(past_key_values, RelayCache)

    if is_relay_cache:
        hidden_states, causal_mask, position_embeddings, position_ids = (
            past_key_values.initial_process(
                hidden_states, causal_mask, position_embeddings, position_ids, self.rotary_emb
            )
        )
        
    for idx, decoder_layer in enumerate(self.layers): 
        if is_relay_cache:
            hidden_states, causal_mask, position_embeddings, position_ids = (
                past_key_values.load_hidden_states(
                    hidden_states, idx, causal_mask, position_embeddings, position_ids,
                )
            )

        if is_relay_cache:
            past_key_values.save_hidden_states(hidden_states, idx)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **flash_attn_kwargs,
        )

        hidden_states = layer_outputs[0]

        if is_relay_cache:
            hidden_states, causal_mask, position_embeddings, position_ids = (
                past_key_values.pre_drop_process(
                    hidden_states, causal_mask, position_embeddings, position_ids, self.rotary_emb
                )
            )
            hidden_states, causal_mask, position_embeddings, position_ids = (
                past_key_values.post_drop_process(
                    hidden_states, causal_mask, position_embeddings, position_ids, self.rotary_emb
                )
            )

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # Reset state after forward pass
    if is_relay_cache and (past_key_values.recompute_label is True or past_key_values.recompute_indices_tensor.numel() > 0):
        # Reset all recompute state (fully tensorized)
        past_key_values.recompute_label = False
        
        # Reset to empty tensors
        empty_long = torch.tensor([], device=past_key_values.device, dtype=torch.long)
        
        past_key_values.recompute_indices_tensor = empty_long
        past_key_values.blank_absolute_indices_tensor = empty_long
        past_key_values.relay_absolute_indices_tensor = empty_long
        past_key_values.blank_relative_indices_tensor = empty_long
        past_key_values.relay_relative_indices_tensor = empty_long
        
        past_key_values.reuse_hidden_layer = None
        past_key_values.drop_label_pre = False
        past_key_values.drop_label_post = False

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def new_update_causal_mask(
    self,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool = False,
):
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)
    using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

    _, query_length = input_tensor.shape[0], input_tensor.shape[1]
    key_value_length = query_length + past_seen_tokens
    
    if input_tensor.shape[1] == 1:
        return None
    
    dtype = input_tensor.dtype
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    
    if using_sliding_window_cache or using_static_cache:
        target_length = past_key_values.get_max_cache_shape()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
        config=self.config,
        past_key_values=past_key_values,
    )
    return causal_mask


@staticmethod
def new_prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    cache_position: torch.Tensor,
    batch_size: int,
    config: Qwen3Config,
    past_key_values: Cache,
):
    """Prepare 4D causal attention mask with RelayCache support."""
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
        )
        diagonal_attend_mask = torch.arange(target_length, device=cache_position.device) > cache_position.reshape(
                -1, 1
        )
        
        if hasattr(past_key_values, "recompute_label") and past_key_values.recompute_label is True:
            recompute_tensor = getattr(past_key_values, 'recompute_indices_tensor', None)
            
            if recompute_tensor is not None and recompute_tensor.numel() > 0:
                if recompute_tensor.device != cache_position.device:
                    recompute_tensor = recompute_tensor.to(cache_position.device)

                diagonal_attend_prefix_mask = torch.arange(
                    target_length, device=cache_position.device
                ) > recompute_tensor.reshape(-1, 1)
            diagonal_attend_mask = torch.cat([diagonal_attend_prefix_mask,diagonal_attend_mask], dim=0)
        
        if config.sliding_window is not None:
            if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                sliding_attend_mask = torch.arange(target_length, device=cache_position.device) <= (
                    cache_position.reshape(-1, 1) - config.sliding_window
                )
                diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
                
        causal_mask *= diagonal_attend_mask
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.shape[-1] > target_length:
                attention_mask = attention_mask[:, :target_length]
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                causal_mask.device
            )
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask





def prepare_inputs_for_generation_qwen3(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    if past_key_values is not None:
        if inputs_embeds is not None: 
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif input_ids.shape[1] != cache_position.shape[0]:
            if (
                hasattr(past_key_values, "recompute_label")
                and past_key_values.recompute_label
            ):
                recompute_tensor = getattr(past_key_values, 'recompute_indices_tensor', None)
                
                if recompute_tensor is not None and recompute_tensor.numel() > 0:
                    recompute_idx = recompute_tensor.to(dtype=torch.long, device=input_ids.device)
                    cache_idx = cache_position.to(dtype=torch.long, device=input_ids.device)
                    input_ids = torch.cat(
                        [
                            input_ids[:, recompute_idx],
                            input_ids[:, cache_idx],
                        ],
                        dim=1,
                    )
            else:
                input_ids = input_ids[:, cache_position]

    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            if (
                hasattr(past_key_values, "recompute_label")
                and past_key_values.recompute_label
            ):
                recompute_tensor = getattr(past_key_values, 'recompute_indices_tensor', None)
                
                if recompute_tensor is not None and recompute_tensor.numel() > 0:
                    index_tensor = recompute_tensor.to(device=input_ids.device, dtype=torch.long).unsqueeze(0)
                    
                    if index_tensor.shape[0] != position_ids.shape[0]:
                         index_tensor = index_tensor.expand(position_ids.shape[0], -1)

                    position_ids = torch.cat(
                        [index_tensor, position_ids[:, -cache_position.shape[0] :]],
                        dim=1,
                    )
                    position_ids = position_ids.clone(memory_format=torch.contiguous_format)
            else:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids.contiguous()}
    
    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs