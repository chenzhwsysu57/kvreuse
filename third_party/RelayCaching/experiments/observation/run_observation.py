#!/usr/bin/env python3
"""Observation Experiment (Paper Section 3)

Runs observation experiments to validate KV cache similarity between
Decoding KV and Prefill KV across layers, with proper RelayCache-based
sweep logic aligned with the profiler architecture.
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

# Add project root and experiments to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
experiments_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, experiments_dir)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from observation.plotting import (
    plot_macro_similarity,
    plot_layer_wise_deviation,
    plot_relative_recovery,
    plot_token_wise_all_metrics,
    plot_inter_layer_correlation,
)
from observation.data_collector import ObservationDataCollector
from observation.layer_selection import auto_select_middle_layer_range

from RelayCaching.utils.profiler_utils import (
    load_model,
    load_dataset,
    normalize_question,
    build_prefix,
    build_suffix,
    ROPE_Module,
    enable_model_attention_prefill_context,
    extract_kv_with_tokens,
    calculate_overall_similarity,
    MAP_PROMPT_TEMPLATE,
    kv_cache_inverse_rope,
    kv_cache_apply_rope,
    extract_KV_from_outputs,
    FILTER_PHRASES,
)
from RelayCaching.utils.relay_utils import RelayCache
from RelayCaching.prompt.prompt_set_registry import PromptSetRegistry
from RelayCaching.tools.reader.readers import JSONLReader
from datasets_self.gsm8k_dataset import universal_data_process
from datasets_self.mmlu_dataset import load_mmlu_observation_records


class ObservationExperiment:
    """Observation experiment class (Paper Section 3).

    Holds model, tokenizer, rope_module, and other resources as instance
    attributes so that methods can create RelayCache objects and run
    model.generate calls directly, matching the profiler architecture.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        dataset_path: str = "./inputs/wikimqa_s.json",
        output_dir: str = "./observation_results",
        st_idx: int = 0,
        data_len: int = 1,
        ctx_num: int = 1,
        layer_lower_threshold: float = 0.99,
        layer_upper_sigma_multiplier: float = 0.5,
        layer_upper_consecutive_count: int = 2,
        layer_upper_min_samples: int = 3,
        right_tail_layers: int = 5,
        layer_selection_metric: str = "value_cosine",
        dataset_type: str = "wikimqa",
        domain: str = None,
        role: str = None,
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.st_idx = st_idx
        self.data_len = data_len
        self.ctx_num = ctx_num
        self.layer_lower_threshold = layer_lower_threshold
        self.layer_upper_sigma_multiplier = layer_upper_sigma_multiplier
        self.layer_upper_consecutive_count = layer_upper_consecutive_count
        self.layer_upper_min_samples = layer_upper_min_samples
        self.right_tail_layers = right_tail_layers
        self.layer_selection_metric = layer_selection_metric
        self.dataset_type = dataset_type
        self.domain = domain
        self.role = role
        self.device_str = device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        print(f"   Device: {self.device}")
        self.model, self.tokenizer = load_model(
            model_name=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device_str,
        )
        self.model.eval()
        enable_model_attention_prefill_context(self.model_name)
        self.num_layers = self.model.config.num_hidden_layers

        self.rope_module = ROPE_Module(self.model_name, self.device)

        # Load dataset
        self._load_dataset()

        # Prompt set
        self.prompt_set = None
        if self.domain:
            self.prompt_set = PromptSetRegistry.get(self.domain)
        elif self.dataset_type in ["gsm8k", "math", "humaneval", "mmlu"]:
            self.prompt_set = PromptSetRegistry.get(self.dataset_type)
        if self.prompt_set and self.role is None:
            self.role = self.prompt_set.get_role()

        # RelayCache precompute state (populated by _prepare_all_caches)
        self._final_cache_with_rope: Optional[DynamicCache] = None
        self._recompute_indices: List[int] = []
        self._addition_indices_list: List[List[int]] = []
        self._ctx_indices_list: List[List[int]] = []
        self._all_layer_hidden_states: Dict[int, List[torch.Tensor]] = {}
        self._information_cache_list: List[DynamicCache] = []
        self._prefix_ids: Optional[torch.Tensor] = None
        self._prefix_len: int = 0
        self._suffix_len: int = 0
        self._partial_len: int = 0
        self._full_input_ids: Optional[torch.Tensor] = None
        self._prefix_cache_with_rope: Optional[DynamicCache] = None
        self._reference_cache: Optional[List[Dict]] = None
        self._summaries: List[str] = []
        self._prefill_kv_list: List[List[Dict]] = []
        self._random_kv_list: List[List[Dict]] = []
        self._independent_kv_list: List[List[Dict]] = []
        self._info_start_positions: List[int] = []
        self._info_lengths: List[int] = []

        self._default_params = {
            "tau_inf": 0.0,
            "prefix_tokens": 0,
            "suffix_tokens": 0,
            "max_new_tokens": 2,
        }

        print(f"   Model: {self.model_name} ({self.num_layers} layers)")
        print(f"   Dataset: {self.dataset_type} ({len(self.dataset)} samples)")
        if self.prompt_set:
            print(f"   PromptSet: {self.domain}, Role: {self.role}")

    def _load_dataset(self):
        """Load and parse dataset based on dataset_type."""
        if self.dataset_type == "wikimqa":
            self.dataset = load_dataset(self.dataset_path)
        elif self.dataset_type == "humaneval":
            if self.dataset_path.endswith(".jsonl"):
                dataset_raw = JSONLReader.parse_file(self.dataset_path)
            else:
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    dataset_raw = json.load(f)
            self.dataset = [{"task": item["prompt"], "test": item["test"]} for item in dataset_raw]
        elif self.dataset_type in ["gsm8k", "math"]:
            if self.dataset_path.endswith(".jsonl"):
                dataset_raw = JSONLReader.parse_file(self.dataset_path)
            else:
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    dataset_raw = json.load(f)
            self.dataset = universal_data_process(dataset_raw)
        elif self.dataset_type == "mmlu":
            self.dataset = load_mmlu_observation_records(self.dataset_path)
        else:
            try:
                self.dataset = load_dataset(self.dataset_path)
            except Exception:
                dataset_raw = JSONLReader.parse_file(self.dataset_path)
                self.dataset = universal_data_process(dataset_raw)

    # ------------------------------------------------------------------
    # MAP Phase: Generate summaries and collect Decoding KV + hidden states
    # ------------------------------------------------------------------

    def _run_map_phase(self) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """Generate summaries (Decoding KV) and collect their KV caches.

        Unlike the profiler, we store Decoding KV as DynamicCache objects
        (without RoPE) directly into self._information_cache_list, and
        collect hidden states into self._all_layer_hidden_states.

        Returns:
            (summaries, summary_token_ids, map_prompt_lens)
        """
        print("\nStep 2: MAP Phase - Generating summaries (Decoding KV)...")
        qa_pair = self.dataset[self.st_idx]

        if self.dataset_type == "wikimqa":
            question = normalize_question(qa_pair["question"])
            contexts = qa_pair["ctxs"][:]
            use_contexts = True
            print(f"   Question: {question}")
        else:
            question = qa_pair["task"]
            contexts = []
            use_contexts = False
            print(f"   Task: {question[:100]}...")

        summaries = []
        summary_token_ids = []
        map_prompt_lens = []
        valid_ctx_count = 0

        # Reset state
        self._information_cache_list = []
        self._all_layer_hidden_states = {}
        recompute_index_list = []

        if use_contexts:
            for ctx_idx, ctx in enumerate(contexts):
                context_text = ctx["text"]
                map_prompt = MAP_PROMPT_TEMPLATE.format(question=question, context=context_text)
                map_prompt_full = build_prefix(self.model_name, map_prompt)
                map_prompt_full = build_suffix(self.model_name, map_prompt_full)

                inputs = self.tokenizer(map_prompt_full, return_tensors="pt", add_special_tokens=False).to(self.device)
                map_prompt_len = inputs["input_ids"].shape[1]

                # Step 1: Generate response tokens
                with torch.no_grad():
                    gen_outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        use_cache=True,
                        do_sample=False,
                        return_dict_in_generate=True,
                    )

                generated_ids = gen_outputs["sequences"][:, :-1]
                summary_ids = generated_ids[:, map_prompt_len:]
                summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                if any(phrase in summary_text.lower() for phrase in FILTER_PHRASES):
                    continue

                # Step 2: Re-run model with full generated ids to get hidden states
                with torch.no_grad():
                    full_outputs = self.model(
                        input_ids=generated_ids,
                        use_cache=True,
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                summaries.append(summary_text)
                summary_token_ids.append(generated_ids[0])
                map_prompt_lens.append(map_prompt_len)

                generated_len = summary_ids.shape[1]
                pos_start = map_prompt_len

                # Extract Decoding KV (without RoPE), matching profiler
                temp_decode_cache = extract_KV_from_outputs(full_outputs.past_key_values, pos_start, generated_len)
                position_ids = torch.tensor(
                    list(range(pos_start, pos_start + generated_len)),
                    device=self.device,
                ).unsqueeze(0)
                temp_kv_no_rope = kv_cache_inverse_rope(temp_decode_cache, position_ids, self.rope_module)
                self._information_cache_list.append(temp_kv_no_rope)

                # Collect hidden states for RelayCache
                for l_idx in range(self.num_layers):
                    if l_idx not in self._all_layer_hidden_states:
                        self._all_layer_hidden_states[l_idx] = []
                    hidden_states_segment = full_outputs.hidden_states[l_idx][
                        :, pos_start : pos_start + generated_len, :
                    ].clone().detach().to(self.device)
                    self._all_layer_hidden_states[l_idx].append(hidden_states_segment)

                recompute_index_list.append(list(range(generated_len)))

                valid_ctx_count += 1
                print(f"   Context {ctx_idx+1}: {generated_len} tokens (Valid #{valid_ctx_count})")
                if valid_ctx_count >= self.ctx_num:
                    break
        else:
            # Non-wikimqa: first generate response tokens, then re-run
            # model() with full ids to get hidden states and information KV
            if self.prompt_set:
                system_prompt = self.prompt_set.get_constraint(self.role)
                if self.dataset_type == "humaneval":
                    user_prompt = self.prompt_set.get_answer_prompt(question=question)
                else:
                    user_prompt = self.prompt_set.get_answer_prompt(question=question, role=self.role)
                map_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                if self.dataset_type == "humaneval":
                    map_prompt = f"{question}\nWrite your full implementation in a Python code block.\n"
                else:
                    map_prompt = f"Q: {question}\nA: Let's think step by step.\n"

            map_prompt_full = build_prefix(self.model_name, map_prompt)
            map_prompt_full = build_suffix(self.model_name, map_prompt_full)

            inputs = self.tokenizer(map_prompt_full, return_tensors="pt", add_special_tokens=False).to(self.device)
            map_prompt_len = inputs["input_ids"].shape[1]

            max_new_tokens = 1024 if self.dataset_type == "humaneval" else 512

            # Step 1: Generate response tokens
            with torch.no_grad():
                gen_outputs = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, use_cache=True,
                    do_sample=False, return_dict_in_generate=True,
                )

            generated_ids = gen_outputs["sequences"][:, :-1]
            response_ids = generated_ids[:, map_prompt_len:]
            response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

            summaries.append(response_text)
            summary_token_ids.append(generated_ids[0])
            map_prompt_lens.append(map_prompt_len)

            response_len = response_ids.shape[1]

            # Step 2: Re-run model with full generated ids to get hidden states
            with torch.no_grad():
                full_outputs = self.model(
                    input_ids=generated_ids,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # Extract hidden states for RelayCache
            for l_idx in range(self.num_layers):
                if l_idx not in self._all_layer_hidden_states:
                    self._all_layer_hidden_states[l_idx] = []
                hidden_states_segment = full_outputs.hidden_states[l_idx][
                    :, map_prompt_len : map_prompt_len + response_len, :
                ].clone().detach().to(self.device)
                self._all_layer_hidden_states[l_idx].append(hidden_states_segment)

            # Extract Decoding KV (without RoPE)
            temp_decode_cache = extract_KV_from_outputs(full_outputs.past_key_values, map_prompt_len, response_len)
            position_ids = torch.tensor(
                list(range(map_prompt_len, map_prompt_len + response_len)),
                device=self.device,
            ).unsqueeze(0)
            temp_kv_no_rope = kv_cache_inverse_rope(temp_decode_cache, position_ids, self.rope_module)
            self._information_cache_list.append(temp_kv_no_rope)

            recompute_index_list.append(list(range(response_len)))
            valid_ctx_count = 1
            print(f"   Generated {response_len} tokens")

        if len(summaries) == 0:
            raise RuntimeError("No valid summaries generated.")

        self._recompute_index_list_raw = recompute_index_list
        self._summaries = summaries
        print(f"   MAP Phase: {len(summaries)} valid summaries")
        return summaries, summary_token_ids, map_prompt_lens

    # ------------------------------------------------------------------
    # Prepare all caches (aligned with profiler._prepare_all_caches)
    # ------------------------------------------------------------------

    def _prepare_all_caches(
        self,
        question: str,
        summaries: List[str],
        summary_token_ids: List[torch.Tensor],
        map_prompt_lens: List[int],
    ):
        """Build final_cache_with_rope, recompute_indices, and all other
        data structures required by RelayCache — matching profiler logic.

        This method constructs:
        - self._prefix_cache_with_rope (prefix KV with RoPE)
        - self._prefix_cache (prefix KV without RoPE)
        - self._final_cache_with_rope (prefix + Decoding KV, with RoPE)
        - self._full_input_ids (complete reduce prompt token ids)
        - self._recompute_indices, _addition_indices_list, _ctx_indices_list
        """
        print("\nPreparing all caches for RelayCache...")

        # --- Build reduce prompt prefix ---
        if self.dataset_type == "wikimqa":
            reduce_prompt_prefix = f"Based on the following information, answer the question:\nQuestion: {question}\n\nInformation:\n"
        elif self.prompt_set:
            decision_role = self.prompt_set.get_decision_role()
            decision_constraint = self.prompt_set.get_decision_constraint()
            decision_few_shot = self.prompt_set.get_decision_few_shot()
            system_prompt = f"{decision_role}\n{decision_constraint}"

            if self.domain in ["humaneval", "mbpp", "ds1000", "humaneval_plus"]:
                user_prompt_before_spatial = (
                    f"{decision_few_shot}{question}\n"
                    f"At the same time, the output of other agents is as follows:\n\n"
                )
            else:
                prefix_text = "Q:\n"
                user_prompt_before_spatial = (
                    f"{decision_few_shot}{prefix_text}{question}\n"
                    f"At the same time, the output of other agents is as follows:\n\n"
                )
            reduce_prompt_prefix = f"{system_prompt}\n\n{user_prompt_before_spatial}"
        else:
            reduce_prompt_prefix = f"Based on the following information, answer the question:\nQuestion: {question}\n\nInformation:\n"

        reduce_prompt_prefix = build_prefix(self.model_name, reduce_prompt_prefix)
        self._prefix_ids = self.tokenizer(
            reduce_prompt_prefix, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self.device)
        self._prefix_len = self._prefix_ids.shape[1]

        # Compute prefix KV cache (with RoPE)
        self._prefix_cache_with_rope = DynamicCache()
        with torch.no_grad():
            self.model(
                self._prefix_ids,
                past_key_values=self._prefix_cache_with_rope,
                return_dict=True,
                use_cache=True,
            )

        # Compute prefix KV cache (without RoPE)
        cache_copy = DynamicCache()
        cache_copy.key_cache = [k.clone() for k in self._prefix_cache_with_rope.key_cache]
        cache_copy.value_cache = [v.clone() for v in self._prefix_cache_with_rope.value_cache]
        if hasattr(self._prefix_cache_with_rope, "_seen_tokens"):
            cache_copy._seen_tokens = self._prefix_cache_with_rope._seen_tokens
        position_ids = torch.tensor(list(range(self._prefix_len)), device=self.device).unsqueeze(0)
        prefix_cache = kv_cache_inverse_rope(cache_copy, position_ids, self.rope_module)

        # --- Build reduce prompt suffix ---
        if self.dataset_type == "wikimqa":
            reduce_prompt_suffix = "\nFinal Answer:"
        elif self.prompt_set:
            if self.domain in ["humaneval", "mbpp", "ds1000", "humaneval_plus"]:
                reduce_prompt_suffix = "\n```python\n"
            else:
                reduce_prompt_suffix = "\nFinal Answer:"
        else:
            reduce_prompt_suffix = "\nAnswer the question based on the above information. Answer concisely.\nAnswer:"

        reduce_prompt_suffix = build_suffix(self.model_name, reduce_prompt_suffix)
        reduce_prompt_suffix_ids = self.tokenizer(
            reduce_prompt_suffix, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(self.device)
        self._suffix_len = reduce_prompt_suffix_ids.shape[1]

        # --- Build information_ids and index lists ---
        current_length = self._prefix_len
        information_ids = torch.empty((1, 0), dtype=torch.int64, device=self.device)

        self._ctx_indices_list = []
        self._addition_indices_list = []

        for i in range(len(self._information_cache_list)):
            recompute_indices_raw = self._recompute_index_list_raw[i]
            self._addition_indices_list.append([x for x in recompute_indices_raw])
            self._recompute_index_list_raw[i] = [
                x + current_length for x in recompute_indices_raw
            ]
            # ctx_indices: [start, end] of this agent's information segment
            info_ids_i = summary_token_ids[i][map_prompt_lens[i]:].unsqueeze(0) \
                if self.dataset_type != "wikimqa" or True else summary_token_ids[i].unsqueeze(0)
            seg_len = info_ids_i.shape[1]
            self._ctx_indices_list.append([current_length, current_length + seg_len - 1])
            current_length += seg_len
            information_ids = torch.cat([information_ids, info_ids_i], dim=-1)

        partial_reduce_prompt_ids = torch.cat([self._prefix_ids, information_ids], dim=-1)
        self._partial_len = partial_reduce_prompt_ids.shape[1]

        self._full_input_ids = torch.cat(
            [partial_reduce_prompt_ids, reduce_prompt_suffix_ids], dim=-1,
        ).contiguous()

        self._recompute_indices = list(range(self._prefix_len, self._partial_len))

        # --- Build final_cache_with_rope ---
        full_past_key_values = DynamicCache()
        for l in range(self.num_layers):
            key_caches = [prefix_cache.key_cache[l].clone()]
            value_caches = [prefix_cache.value_cache[l].clone()]

            for info_cache in self._information_cache_list:
                key_caches.append(info_cache.key_cache[l].clone())
                value_caches.append(info_cache.value_cache[l].clone())

            full_past_key_values.key_cache.append(torch.cat(key_caches, dim=-2))
            full_past_key_values.value_cache.append(torch.cat(value_caches, dim=-2))

        position_ids = torch.tensor(
            list(range(self._partial_len)), device=self.device,
        ).unsqueeze(0)
        self._final_cache_with_rope = kv_cache_apply_rope(
            full_past_key_values, position_ids, self.rope_module,
        )

        print(f"   Cache preparation complete. Total input length: {self._full_input_ids.shape[1]}")

    # ------------------------------------------------------------------
    # Gold baseline (aligned with profiler._run_gold_baseline)
    # ------------------------------------------------------------------

    def _run_gold_baseline(self) -> List[Dict]:
        """Run Full Prefill baseline: prefix KV only + model.generate.

        Returns:
            reference_cache: suffix KV extracted from the gold baseline run,
                             as a list of dicts [{key, value}] per layer.
        """
        print("   Running GOLD BASELINE (Full Prefill)...")

        gold_cache = DynamicCache()
        for l in range(self.num_layers):
            gold_cache.key_cache.append(
                self._final_cache_with_rope.key_cache[l][:, :, :self._prefix_len, :].clone()
            )
            gold_cache.value_cache.append(
                self._final_cache_with_rope.value_cache[l][:, :, :self._prefix_len, :].clone()
            )

        with torch.no_grad():
            outputs = self.model.generate(
                self._full_input_ids,
                max_new_tokens=self._default_params.get("max_new_tokens", 2),
                use_cache=True,
                do_sample=False,
                past_key_values=gold_cache,
                return_dict_in_generate=True,
                output_attentions=False,
            )

        reference_cache = extract_kv_with_tokens(
            outputs, self._partial_len, self._suffix_len,
        )
        print("   GOLD BASELINE complete.")
        return reference_cache

    # ------------------------------------------------------------------
    # No-repair baseline (aligned with profiler._run_no_repair_baseline)
    # ------------------------------------------------------------------

    def _run_no_repair_baseline(self, mode: str = "high") -> Tuple[List[Dict], Dict[str, float]]:
        """Run no-repair baseline using RelayCache with det_layer set to
        disable all repair logic, matching the profiler approach.

        Args:
            mode: "low" (det_layer = num_layers + 1, profile_lower=True)
                  or "high" (det_layer = -1, profile_upper=True)

        Returns:
            (no_repair_cache_data, no_repair_similarity)
        """
        print(f"   Running NO REPAIR BASELINE ({mode} mode)...")

        profile_lower = (mode == "low")
        profile_upper = (mode == "high")

        current_layer_hidden_states = self._all_layer_hidden_states[0]

        no_repair_cache = RelayCache(
            self._final_cache_with_rope,
            recompute_indices=self._recompute_indices,
            blank_absolute_indices=[],
            blank_relative_indices=[],
            relay_absolute_indices=self._recompute_indices,
            relay_relative_indices=list(range(len(self._recompute_indices))),
            recompute_label=True,
            det_layer=self.num_layers + 1 if profile_lower else -1,
            end_layer=self.num_layers,
            tau_dev=0.0,
            tau_inf=self._default_params.get("tau_inf", 0.0),
            prefix_tokens=self._default_params.get("prefix_tokens", 0),
            suffix_tokens=self._default_params.get("suffix_tokens", 0),
            reuse_hidden_layer=0,
            store_hidden_layer=None,
            execution_mode="relaycaching",
            additional_indices=self._addition_indices_list,
            relay_chunk_indices=self._ctx_indices_list,
            profile_lower_layer=profile_lower,
            profile_upper_layer=profile_upper,
            reusable_hidden_states_list=current_layer_hidden_states,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                self._full_input_ids,
                max_new_tokens=self._default_params.get("max_new_tokens", 2),
                use_cache=True,
                do_sample=False,
                past_key_values=no_repair_cache,
                return_dict_in_generate=True,
                output_attentions=False,
            )

        no_repair_cache_data = extract_kv_with_tokens(
            outputs, self._partial_len, self._suffix_len,
        )

        no_repair_similarity = calculate_overall_similarity(
            self._reference_cache, no_repair_cache_data,
        )
        print(f"   NO REPAIR similarity: "
              f"key_cos={no_repair_similarity.get('key_cosine_similarity', 0):.4f}, "
              f"value_cos={no_repair_similarity.get('value_cosine_similarity', 0):.4f}")

        return no_repair_cache_data, no_repair_similarity

    # ------------------------------------------------------------------
    # Sweep experiment (aligned with profiler._run_experiment_reuse)
    # ------------------------------------------------------------------

    def _run_sweep(
        self,
        mode: str,
        sweep_values: List[int],
        fixed_start_layer: Optional[int] = None,
    ) -> Dict[int, List[Dict]]:
        """Run RelayCache-based sweep, matching profiler logic.

        For each sweep value, creates a RelayCache with the corresponding
        parameters and runs model.generate, then extracts suffix KV.

        Args:
            mode: "low" or "high"
            sweep_values: list of layer indices to sweep
            fixed_start_layer: for "high" mode, fix the reuse_hidden_layer

        Returns:
            {sweep_value: extracted_kv_data}
        """
        print(f"\n   Running SWEEP ({mode} mode) over {len(sweep_values)} values...")

        profile_lower = (mode == "low")
        profile_upper = (mode == "high")

        all_generated_data = {}

        for layer_val in sweep_values:
            if profile_lower:
                det_layer = layer_val
                end_layer_val = self.num_layers
                reuse_layer_idx = layer_val
            elif profile_upper:
                det_layer = layer_val
                end_layer_val = self.num_layers
                reuse_layer_idx = fixed_start_layer if fixed_start_layer is not None else layer_val
                if fixed_start_layer is not None and layer_val < fixed_start_layer:
                    continue
            else:
                det_layer = layer_val
                end_layer_val = self.num_layers
                reuse_layer_idx = layer_val

            current_layer_hidden_states = self._all_layer_hidden_states[reuse_layer_idx]

            relay_cache = RelayCache(
                self._final_cache_with_rope,
                recompute_indices=self._recompute_indices,
                blank_absolute_indices=[],
                blank_relative_indices=[],
                relay_absolute_indices=self._recompute_indices,
                relay_relative_indices=list(range(len(self._recompute_indices))),
                recompute_label=True,
                det_layer=det_layer,
                end_layer=end_layer_val,
                tau_dev=0.0,
                tau_inf=self._default_params.get("tau_inf", 0.0),
                prefix_tokens=self._default_params.get("prefix_tokens", 0),
                suffix_tokens=self._default_params.get("suffix_tokens", 0),
                reuse_hidden_layer=reuse_layer_idx,
                store_hidden_layer=None,
                execution_mode="relaycaching",
                additional_indices=self._addition_indices_list,
                relay_chunk_indices=self._ctx_indices_list,
                profile_lower_layer=profile_lower,
                profile_upper_layer=profile_upper,
                reusable_hidden_states_list=current_layer_hidden_states,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    self._full_input_ids,
                    max_new_tokens=self._default_params.get("max_new_tokens", 2),
                    use_cache=True,
                    do_sample=False,
                    past_key_values=relay_cache,
                    return_dict_in_generate=True,
                    output_attentions=False,
                )

            generated_kv = extract_kv_with_tokens(
                outputs, self._partial_len, self._suffix_len,
            )
            all_generated_data[layer_val] = generated_kv
            print(f"     Layer {layer_val}: done")

        return all_generated_data

    def _analyze_sweep_results(
        self,
        mode: str,
        sweep_values: List[int],
        generated_data: Dict[int, List[Dict]],
    ) -> Dict[str, List[float]]:
        """Analyze sweep results by comparing with reference_cache."""
        print(f"\n   Analyzing {mode.upper()} Bound Results...")

        if self._reference_cache is None:
            raise RuntimeError("Reference cache (gold baseline) missing.")

        results_lists = {
            "key_cosine": [],
            "value_cosine": [],
            "key_norm": [],
            "value_norm": [],
        }

        for val in sweep_values:
            recomputed_kv = generated_data.get(val)

            if recomputed_kv is None:
                for key in results_lists:
                    results_lists[key].append(float('nan'))
                continue

            overall_metrics = calculate_overall_similarity(
                self._reference_cache, recomputed_kv,
            )

            if not overall_metrics:
                for key in results_lists:
                    results_lists[key].append(float('nan'))
            else:
                results_lists["key_cosine"].append(
                    overall_metrics.get("key_cosine_similarity", float('nan'))
                )
                results_lists["value_cosine"].append(
                    overall_metrics.get("value_cosine_similarity", float('nan'))
                )
                results_lists["key_norm"].append(
                    overall_metrics.get("key_norm_similarity", float('nan'))
                )
                results_lists["value_norm"].append(
                    overall_metrics.get("value_norm_similarity", float('nan'))
                )

        return results_lists

    # ------------------------------------------------------------------
    # Prefill KV collection (for observation layer-wise comparison)
    # ------------------------------------------------------------------

    def _collect_prefill_kv(
        self,
        question: str,
        summaries: List[str],
        summary_token_ids: List[torch.Tensor],
        map_prompt_lens: List[int],
    ) -> Tuple[List[List[Dict]], List[List[Dict]], List[List[Dict]], List[int], List[int]]:
        """Prefill summaries in the reduce context and collect prefill KV.

        Also collects Random KV and Independent KV baselines (original Step 3.5 / 3.6).

        Returns:
            (prefill_kv_list, random_kv_list, independent_kv_list, info_start_positions, info_lengths)
        """
        print("\nStep 3: REUSE Phase - Prefilling summaries (Prefill KV)...")

        info_start_positions = []
        info_lengths = []
        all_info_ids_list = []
        prefill_kv_list = []
        random_kv_list = []
        independent_kv_list = []

        if self.prompt_set:
            decision_role = self.prompt_set.get_decision_role()
            decision_constraint = self.prompt_set.get_decision_constraint()
            decision_few_shot = self.prompt_set.get_decision_few_shot()
            system_prompt = f"{decision_role}\n{decision_constraint}"

            user_prompt_before_spatial = (
                f"{decision_few_shot}Q:\n{question}\n"
                f"At the same time, the output of other agents is as follows:\n\n"
            )
            reduce_prompt_before_spatial = f"{system_prompt}\n\n{user_prompt_before_spatial}"
            reduce_prompt_before_spatial = build_prefix(self.model_name, reduce_prompt_before_spatial)

            prefix_before_spatial_ids = self.tokenizer(
                reduce_prompt_before_spatial, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(self.device)
            current_pos = prefix_before_spatial_ids.shape[1]

            spatial_segments_ids = []
            for agent_idx, summary_text in enumerate(summaries):
                agent_role = self.role if self.role else "Agent"
                agent_header = f"Agent {agent_idx + 1}, role is {agent_role}, output is:\n\n"
                header_ids = self.tokenizer(agent_header, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                spatial_segments_ids.append(header_ids)
                current_pos += header_ids.shape[1]

                summary_start = current_pos
                # Use MAP-phase token ids (same as Decoding KV), not decode→re-tokenize(summary_text)
                if agent_idx < len(summary_token_ids) and agent_idx < len(map_prompt_lens):
                    summary_ids = summary_token_ids[agent_idx][map_prompt_lens[agent_idx] :].unsqueeze(0).to(self.device)
                else:
                    summary_ids = self.tokenizer(
                        summary_text, return_tensors="pt", add_special_tokens=False,
                    ).input_ids.to(self.device)
                spatial_segments_ids.append(summary_ids)
                summary_len = summary_ids.shape[1]
                current_pos += summary_len

                info_start_positions.append(summary_start)
                info_lengths.append(summary_len)
                all_info_ids_list.append(summary_ids)

                suffix_ids = self.tokenizer("\n\n", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                spatial_segments_ids.append(suffix_ids)
                current_pos += suffix_ids.shape[1]

            after_spatial_ids = self.tokenizer("\n", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

            if self.dataset_type == "humaneval":
                output_format = "Write your final implementation. Use a Python code block.\n```python\n"
            else:
                output_format = "The answer is "

            reduce_suffix_ids = self.tokenizer(
                build_suffix(self.model_name, output_format),
                return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(self.device)
            full_reduce_ids = torch.cat(
                [prefix_before_spatial_ids] + spatial_segments_ids + [after_spatial_ids, reduce_suffix_ids],
                dim=1,
            )
        else:
            if self.dataset_type == "humaneval":
                reduce_prompt_prefix = f"Based on the function signature below and the provided implementations, write the final implementation.\n{question}\nImplementations:\n\n"
                reduce_prompt_suffix = "Write your final implementation.\n```python\n"
            elif self.dataset_type in ["gsm8k", "math"]:
                reduce_prompt_prefix = f"Answer the question based on the given solutions.\nQuestion:{question}\nSolutions:\n\n"
                reduce_prompt_suffix = f"Answer the question within 5 words.\nQuestion:{question}\nAnswer:"
            else:
                reduce_prompt_prefix = f"Answer the question based on the given informations.\nQuestion:{question}\nInfos:\n\n"
                reduce_prompt_suffix = f"Answer the question within 5 words.\nQuestion:{question}\nAnswer:"

            reduce_prompt_prefix = build_prefix(self.model_name, reduce_prompt_prefix)
            reduce_prefix_ids = self.tokenizer(reduce_prompt_prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            current_pos = reduce_prefix_ids.shape[1]

            for ctx_idx, (summary_ids_full, map_prompt_len) in enumerate(zip(summary_token_ids, map_prompt_lens)):
                summary_only_ids = summary_ids_full[map_prompt_len:].unsqueeze(0)
                all_info_ids_list.append(summary_only_ids)
                info_start_positions.append(current_pos)
                info_lengths.append(summary_only_ids.shape[1])
                current_pos += summary_only_ids.shape[1]

            reduce_prompt_suffix = build_suffix(self.model_name, reduce_prompt_suffix)
            reduce_suffix_ids = self.tokenizer(reduce_prompt_suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            full_reduce_ids = torch.cat([reduce_prefix_ids] + all_info_ids_list + [reduce_suffix_ids], dim=1)

        with torch.no_grad():
            reduce_outputs = self.model(input_ids=full_reduce_ids, use_cache=True, return_dict=True)

        for ctx_idx, (start_pos, summary_len) in enumerate(zip(info_start_positions, info_lengths)):
            temp_prefill_cache = extract_KV_from_outputs(reduce_outputs.past_key_values, start_pos, summary_len)
            position_ids = torch.tensor(
                list(range(start_pos, start_pos + summary_len)),
                device=self.device,
            ).unsqueeze(0)
            prefill_kv_no_rope = kv_cache_inverse_rope(temp_prefill_cache, position_ids, self.rope_module)

            prefill_kv = []
            for layer_idx in range(self.num_layers):
                prefill_kv.append({
                    'key': prefill_kv_no_rope.key_cache[layer_idx].squeeze(0),
                    'value': prefill_kv_no_rope.value_cache[layer_idx].squeeze(0),
                })
            prefill_kv_list.append(prefill_kv)

        # Random KV baseline (original Step 3.5): same length as prefill summary tokens
        print("\nStep 3.5: Random KV baseline...")
        vocab_size = self.tokenizer.vocab_size
        safe_vocab_start = 1000
        safe_vocab_end = vocab_size - 1000
        for ctx_idx, summary_ids in enumerate(all_info_ids_list):
            if summary_ids.dim() == 1:
                summary_ids = summary_ids.unsqueeze(0)
            summary_len = summary_ids.shape[1]
            random_token_ids = torch.randint(
                low=safe_vocab_start,
                high=safe_vocab_end,
                size=(1, summary_len),
                device=self.device,
            )
            with torch.no_grad():
                random_outputs = self.model(input_ids=random_token_ids, use_cache=True, return_dict=True)

            temp_random_cache = extract_KV_from_outputs(random_outputs.past_key_values, 0, summary_len)
            position_ids = torch.tensor(list(range(summary_len)), device=self.device).unsqueeze(0)
            random_kv_no_rope = kv_cache_inverse_rope(temp_random_cache, position_ids, self.rope_module)

            random_kv = []
            for layer_idx in range(self.num_layers):
                random_kv.append({
                    'key': random_kv_no_rope.key_cache[layer_idx].squeeze(0),
                    'value': random_kv_no_rope.value_cache[layer_idx].squeeze(0),
                })
            random_kv_list.append(random_kv)

        # Independent KV baseline: prefill the *same token ids* as in reduce/prefill
        print("\nStep 3.6: Independent KV baseline...")
        for ctx_idx, summary_ids in enumerate(all_info_ids_list):
            if summary_ids.dim() == 1:
                summary_ids = summary_ids.unsqueeze(0)
            summary_ids = summary_ids.to(self.device)

            with torch.no_grad():
                independent_outputs = self.model(input_ids=summary_ids, use_cache=True, return_dict=True)

            summary_len = summary_ids.shape[1]
            temp_independent_cache = extract_KV_from_outputs(independent_outputs.past_key_values, 0, summary_len)
            position_ids = torch.tensor(list(range(summary_len)), device=self.device).unsqueeze(0)
            independent_kv_no_rope = kv_cache_inverse_rope(temp_independent_cache, position_ids, self.rope_module)

            independent_kv = []
            for layer_idx in range(self.num_layers):
                independent_kv.append({
                    'key': independent_kv_no_rope.key_cache[layer_idx].squeeze(0),
                    'value': independent_kv_no_rope.value_cache[layer_idx].squeeze(0),
                })
            independent_kv_list.append(independent_kv)

        print(f"   Prefill KV: {len(prefill_kv_list)} caches")
        print(f"   Random KV: {len(random_kv_list)} caches")
        print(f"   Independent KV: {len(independent_kv_list)} caches")

        return prefill_kv_list, random_kv_list, independent_kv_list, info_start_positions, info_lengths

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the full observation experiment pipeline."""
        print("=" * 80)
        print("Observation Experiment (Paper Section 3)")
        print("=" * 80)

        # Step 2: MAP Phase — generate summaries, collect Decoding KV + hidden states
        summaries, summary_token_ids, map_prompt_lens = self._run_map_phase()

        # Extract question
        qa_pair = self.dataset[self.st_idx]
        if self.dataset_type == "wikimqa":
            question = normalize_question(qa_pair["question"])
        else:
            question = qa_pair["task"]

        # Step 3: Prepare all caches for RelayCache (aligned with profiler)
        self._prepare_all_caches(question, summaries, summary_token_ids, map_prompt_lens)

        # Step 4: Gold baseline — Full Prefill suffix KV as reference
        self._reference_cache = self._run_gold_baseline()

        # Step 5: No-repair baseline — Decoding KV directly written, no repair
        _, no_repair_sim_low = self._run_no_repair_baseline(mode="low")
        _, no_repair_sim_high = self._run_no_repair_baseline(mode="high")

        # Step 6: Sweep — real RelayCache-based sweep over layers
        sweep_values = list(range(self.num_layers))
        print("\nStep 6: Running Layer Sweep...")

        # Low bound sweep (profile_lower_layer=True)
        low_data = self._run_sweep(mode="low", sweep_values=sweep_values)
        results_low = self._analyze_sweep_results("low", sweep_values, low_data)

        # High bound sweep (profile_upper_layer=True)
        fixed_start_layer = 0
        high_data = self._run_sweep(mode="high", sweep_values=sweep_values, fixed_start_layer=fixed_start_layer)
        results_high = self._analyze_sweep_results("high", sweep_values, high_data)

        # Step 7: Collect Prefill KV for observation (Decoding KV vs Prefill KV)
        print("\nStep 7: Collecting Prefill KV for observation...")
        prefill_kv_list, random_kv_list, independent_kv_list, info_start_positions, info_lengths = self._collect_prefill_kv(
            question, summaries, summary_token_ids, map_prompt_lens,
        )
        self._prefill_kv_list = prefill_kv_list
        self._random_kv_list = random_kv_list
        self._independent_kv_list = independent_kv_list
        self._info_start_positions = info_start_positions
        self._info_lengths = info_lengths

        # Compute Decoding KV vs Prefill KV similarities
        print("\nStep 8: Computing Decoding vs Prefill similarities...")

        # Build decoding_kv_list from _information_cache_list for comparison
        decoding_kv_list = []
        for info_cache in self._information_cache_list:
            dec_kv = []
            for layer_idx in range(self.num_layers):
                dec_kv.append({
                    'key': info_cache.key_cache[layer_idx].squeeze(0),
                    'value': info_cache.value_cache[layer_idx].squeeze(0),
                })
            decoding_kv_list.append(dec_kv)

        all_similarities = []
        for ctx_idx, (dec_kv, pre_kv, start_pos, summary_len) in enumerate(
            zip(decoding_kv_list, prefill_kv_list, info_start_positions, info_lengths)
        ):
            reduce_position_ids = torch.tensor(
                list(range(start_pos, start_pos + summary_len)),
                device=self.device,
            ).unsqueeze(0)

            decoding_cache = DynamicCache()
            for layer_kv in dec_kv:
                decoding_cache.update(
                    layer_kv['key'].unsqueeze(0), layer_kv['value'].unsqueeze(0),
                    layer_idx=len(decoding_cache.key_cache),
                )
            decoding_cache = kv_cache_apply_rope(decoding_cache, reduce_position_ids, self.rope_module)

            prefill_cache = DynamicCache()
            for layer_kv in pre_kv:
                prefill_cache.update(
                    layer_kv['key'].unsqueeze(0), layer_kv['value'].unsqueeze(0),
                    layer_idx=len(prefill_cache.key_cache),
                )
            prefill_cache = kv_cache_apply_rope(prefill_cache, reduce_position_ids, self.rope_module)

            dec_with_rope = [
                {'key': decoding_cache.key_cache[l].squeeze(0), 'value': decoding_cache.value_cache[l].squeeze(0)}
                for l in range(self.num_layers)
            ]
            pre_with_rope = [
                {'key': prefill_cache.key_cache[l].squeeze(0), 'value': prefill_cache.value_cache[l].squeeze(0)}
                for l in range(self.num_layers)
            ]

            similarity = calculate_overall_similarity(dec_with_rope, pre_with_rope)
            all_similarities.append(similarity)

        avg_similarity = {k: np.mean([s[k] for s in all_similarities]) for k in all_similarities[0]}
        print(f"   Avg Key Cosine: {avg_similarity['key_cosine_similarity']:.4f}")
        print(f"   Avg Value Cosine: {avg_similarity['value_cosine_similarity']:.4f}")

        # Step 9: Collect detailed data for plotting
        print("\nStep 9: Collecting detailed data...")

        ctx_idx = 0
        start_pos = info_start_positions[ctx_idx]
        summary_len = info_lengths[ctx_idx]
        reduce_position_ids = torch.tensor(
            list(range(start_pos, start_pos + summary_len)),
            device=self.device,
        ).unsqueeze(0)

        def apply_rope_to_kv_list(kv_list_item):
            cache = DynamicCache()
            for layer_kv in kv_list_item:
                cache.update(
                    layer_kv['key'].unsqueeze(0), layer_kv['value'].unsqueeze(0),
                    layer_idx=len(cache.key_cache),
                )
            return kv_cache_apply_rope(cache, reduce_position_ids, self.rope_module)

        decoding_cache = apply_rope_to_kv_list(decoding_kv_list[ctx_idx])
        prefill_cache = apply_rope_to_kv_list(prefill_kv_list[ctx_idx])
        random_cache = apply_rope_to_kv_list(random_kv_list[ctx_idx])
        independent_cache = apply_rope_to_kv_list(independent_kv_list[ctx_idx])

        def cache_to_tuples(cache):
            return [(cache.key_cache[l], cache.value_cache[l]) for l in range(self.num_layers)]

        decoding_kv_tensors = cache_to_tuples(decoding_cache)
        prefill_kv_tensors = cache_to_tuples(prefill_cache)
        random_kv_tensors = cache_to_tuples(random_cache)
        independent_kv_tensors = cache_to_tuples(independent_cache)

        collector = ObservationDataCollector(num_layers=self.num_layers, save_dir=str(self.output_dir / "data"))
        collector.collect_layer_wise_similarity(decoding_kv_tensors, prefill_kv_tensors)
        collector.collect_token_wise_similarity(decoding_kv_tensors, prefill_kv_tensors)
        collector.collect_macro_similarity(
            decoding_kv_tensors, prefill_kv_tensors,
            random_kv=random_kv_tensors, independent_kv=independent_kv_tensors,
        )

        # Token / sequence alignment check (position i = same summary token index after ROPE)
        dk, _ = decoding_kv_tensors[0]
        pk, _ = prefill_kv_tensors[0]
        rk, _ = random_kv_tensors[0]
        ik, _ = independent_kv_tensors[0]
        map_summary_len = (
            summary_token_ids[ctx_idx][map_prompt_lens[ctx_idx] :].shape[0]
            if ctx_idx < len(summary_token_ids) and ctx_idx < len(map_prompt_lens)
            else -1
        )
        print(f"   Token alignment: MAP summary len={map_summary_len}, "
              f"reduce info_len={summary_len}, KV seq lens "
              f"dec={dk.shape[-2]}, pre={pk.shape[-2]}, rnd={rk.shape[-2]}, indep={ik.shape[-2]}")
        if map_summary_len >= 0 and map_summary_len != summary_len:
            print(f"   WARNING: MAP summary token count != reduce summary segment length")
        if not (dk.shape[-2] == pk.shape[-2] == rk.shape[-2] == ik.shape[-2]):
            print(f"   WARNING: KV sequence lengths differ — macro uses min-length alignment")

        # Auto-select middle layer range
        auto_lower_bound, auto_upper_bound = auto_select_middle_layer_range(
            layer_similarities=collector.layer_similarities,
            lower_threshold=self.layer_lower_threshold,
            upper_sigma_multiplier=self.layer_upper_sigma_multiplier,
            upper_consecutive_count=self.layer_upper_consecutive_count,
            upper_min_samples=self.layer_upper_min_samples,
            right_tail_layers=self.right_tail_layers,
            metric=self.layer_selection_metric,
        )

        print(f"\nAuto-selected layer range: [{auto_lower_bound}, {auto_upper_bound}] (inclusive)")

        collector.save_data()

        macro_sims = collector.macro_similarities
        if macro_sims:
            print("\n   Macro similarity (vs Prefill, for Fig. 1):")
            for label, key in (
                ("Decoding KV", "decoding_vs_prefill"),
                ("Random KV", "random_vs_prefill"),
                ("Independent KV", "independent_vs_prefill"),
            ):
                m = macro_sims.get(key, {})
                if m:
                    print(f"      {label}: key_cos={m.get('key_cosine', 0):.4f}, "
                          f"value_cos={m.get('value_cosine', 0):.4f}, "
                          f"key_norm={m.get('key_norm', 0):.4f}, "
                          f"value_norm={m.get('value_norm', 0):.4f}")

        # Step 10: Generate plots
        print("\nStep 10: Generating plots...")

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        plot_macro_similarity(
            macro_similarities=collector.macro_similarities,
            save_path=str(plot_dir / "macro_similarity.pdf"),
        )

        short_model = self.model_name.split('/')[-1]

        if collector.layer_similarities:
            plot_layer_wise_deviation(
                layer_similarities=collector.layer_similarities,
                model_name=short_model,
                L_start=auto_lower_bound,
                L_end=auto_upper_bound,
                save_path=str(plot_dir / "layer_wise_deviation.pdf"),
            )

        if collector.token_similarities:
            first_layer = next(iter(collector.token_similarities.values()))
            suffix_start = len(first_layer.get('key_cosine', [])) - 4
            plot_token_wise_all_metrics(
                token_similarities_by_layer=collector.token_similarities,
                model_name=short_model,
                avg_layer_range=(auto_lower_bound, auto_upper_bound),
                prefix_len=3,
                suffix_start=suffix_start,
                save_path=str(plot_dir / "token_wise_all_metrics.pdf"),
            )

        # Relative recovery (exp_plot: lower/upper bound sweep, value_cosine)
        if results_low.get("value_cosine") and results_high.get("value_cosine"):
            plot_relative_recovery(
                lower_sims=results_low["value_cosine"],
                upper_sims=results_high["value_cosine"],
                L_start=auto_lower_bound,
                L_end=auto_upper_bound,
                save_path=str(plot_dir / "relative_recovery.pdf"),
            )

        # Fig. 5: Inter-layer correlation
        if len(collector.token_similarities) > 1:
            layer_devs = {}
            for k, v in collector.token_similarities.items():
                if isinstance(v, dict) and 'value_cosine' in v:
                    layer_devs[int(k)] = np.array(v['value_cosine'])
            if len(layer_devs) > 1:
                plot_inter_layer_correlation(
                    layer_deviations=layer_devs,
                    save_path=str(plot_dir / "inter_layer_correlation.pdf"),
                )

        print(f"\nAll plots saved to: {plot_dir}")

        # Step 11: Save report
        report = {
            "experiment_config": {
                "model_name": self.model_name,
                "dataset_path": self.dataset_path,
                "dataset_type": self.dataset_type,
                "st_idx": self.st_idx,
                "data_len": self.data_len,
                "num_layers": self.num_layers,
            },
            "auto_selected_layer_range": {
                "lower_bound": auto_lower_bound,
                "upper_bound": auto_upper_bound,
            },
            "average_similarity": avg_similarity,
            "no_repair_similarity_low": no_repair_sim_low,
            "no_repair_similarity_high": no_repair_sim_high,
            "sweep_results_low": {
                "sweep_values": sweep_values,
                "key_cosine": results_low["key_cosine"],
                "value_cosine": results_low["value_cosine"],
            },
            "sweep_results_high": {
                "sweep_values": sweep_values,
                "key_cosine": results_high["key_cosine"],
                "value_cosine": results_high["value_cosine"],
            },
            "timestamp": datetime.now().isoformat(),
        }

        report_path = self.output_dir / "experiment_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print("Observation Experiment Completed!")
        print(f"{'='*80}")

        return report


def load_and_plot_from_saved_data(
    data_path: str = "./observation_results/data/observation_data.json",
    output_dir: str = "./observation_results/plots",
):
    """Load and plot from previously saved data file."""
    print("Loading data from:", data_path)

    data = ObservationDataCollector.load_data(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    macro_data = data.get('macro_similarities', {})
    if macro_data:
        plot_macro_similarity(macro_similarities=macro_data, save_path=str(output_dir / "macro_similarity.pdf"))

    report_path = Path(data_path).parent.parent / "experiment_report.json"
    report = {}
    if report_path.exists():
        with open(report_path, encoding='utf-8') as f:
            report = json.load(f)

    layer_range = report.get("auto_selected_layer_range", {})
    L_start = layer_range.get("lower_bound", 3)
    L_end = layer_range.get("upper_bound", 21)
    model_name = (
        report.get("experiment_config", {}).get("model_name", "Model").split("/")[-1]
    )

    layer_sims = data.get('layer_similarities', [])
    if layer_sims:
        plot_layer_wise_deviation(
            layer_similarities=layer_sims,
            L_start=L_start,
            L_end=L_end,
            save_path=str(output_dir / "layer_wise_deviation.pdf"),
        )

    token_sims = data.get('token_similarities', {})
    if token_sims:
        first_layer = next(iter(token_sims.values()))
        suffix_start = len(first_layer.get('key_cosine', [])) - 4
        plot_token_wise_all_metrics(
            token_similarities_by_layer=token_sims,
            model_name=model_name,
            avg_layer_range=(L_start, L_end),
            prefix_len=3,
            suffix_start=suffix_start,
            save_path=str(output_dir / "token_wise_all_metrics.pdf"),
        )

        if len(token_sims) > 1:
            layer_devs = {}
            for k, v in token_sims.items():
                if isinstance(v, dict) and 'value_cosine' in v:
                    layer_devs[int(k)] = np.array(v['value_cosine'])
            if len(layer_devs) > 1:
                plot_inter_layer_correlation(
                    layer_deviations=layer_devs,
                    save_path=str(output_dir / "inter_layer_correlation.pdf"),
                )

    sweep_low = report.get("sweep_results_low", {}).get("value_cosine")
    sweep_high = report.get("sweep_results_high", {}).get("value_cosine")
    if sweep_low and sweep_high:
        plot_relative_recovery(
            lower_sims=sweep_low,
            upper_sims=sweep_high,
            L_start=L_start,
            L_end=L_end,
            save_path=str(output_dir / "relative_recovery.pdf"),
        )

    print(f"Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Observation Experiment (Paper Section 3)",
    )

    parser.add_argument("--mode", type=str, default="run", choices=["run", "plot"])
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--data", type=str, default="./datasets_self/wikimqa_s.json")
    parser.add_argument("--dataset-type", type=str, default="wikimqa",
                        choices=["wikimqa", "gsm8k", "math", "humaneval", "mmlu"])
    parser.add_argument("--output", type=str, default="./observation_results")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--st-idx", type=int, default=0)
    parser.add_argument("--data-len", type=int, default=1)
    parser.add_argument("--ctx-num", type=int, default=1)

    parser.add_argument("--layer-lower-threshold", type=float, default=0.99,
                        help="τ_low: L_start threshold (deepest layer with sim >= value).")
    parser.add_argument("--layer-upper-sigma-multiplier", type=float, default=0.5,
                        help="λ: L_end stable-band multiplier (mean - λ·σ).")
    parser.add_argument("--layer-upper-consecutive-count", type=int, default=2,
                        help="C: consecutive stable layers for L_end.")
    parser.add_argument("--layer-upper-min-samples", type=int, default=3,
                        help="N_min: min rise samples before L_end detection.")
    parser.add_argument("--right-tail-layers", type=int, default=5,
                        help="T: trailing layers for deep stable-region stats.")
    parser.add_argument("--layer-selection-metric", type=str, default="value_cosine",
                        choices=["key_cosine", "value_cosine", "key_norm", "value_norm"])
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--role", type=str, default=None)
    parser.add_argument("--data-file", type=str, default="./observation_results/data/observation_data.json")

    args = parser.parse_args()

    if args.mode == "run":
        experiment = ObservationExperiment(
            model_name=args.model,
            dataset_path=args.data,
            output_dir=args.output,
            st_idx=args.st_idx,
            data_len=args.data_len,
            ctx_num=args.ctx_num,
            layer_lower_threshold=args.layer_lower_threshold,
            layer_upper_sigma_multiplier=args.layer_upper_sigma_multiplier,
            layer_upper_consecutive_count=args.layer_upper_consecutive_count,
            layer_upper_min_samples=args.layer_upper_min_samples,
            right_tail_layers=args.right_tail_layers,
            layer_selection_metric=args.layer_selection_metric,
            dataset_type=args.dataset_type,
            domain=args.domain,
            role=args.role,
            device=args.device,
        )
        experiment.run()
    elif args.mode == "plot":
        load_and_plot_from_saved_data(
            data_path=args.data_file,
            output_dir=args.output + "/plots",
        )


if __name__ == "__main__":
    main()
