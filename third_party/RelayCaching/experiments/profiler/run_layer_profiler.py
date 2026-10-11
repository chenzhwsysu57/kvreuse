import argparse
import json
import logging
import sys
import time
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

# Optional: SciencePlots if available
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

_GRID_STYLE = {"color": "#E0E0E0", "linestyle": "-", "linewidth": 0.8}

current_file_path = Path(__file__).resolve()
# Get 'RelayCaching' root directory (assuming structure: RelayCaching/experiments/profiler/run_layer_profiler.py)
project_root = current_file_path.parents[2] 
# Add project root to sys.path so we can import as `from RelayCaching.utils...`
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Local Project Imports ---
# Try absolute import first (Best Practice), fallback to relative if running as script
try:
    from RelayCaching.utils.relay_utils import RelayCache
except ImportError:
    # Fallback: assume relay_utils is in the same directory
    try:
        from relay_utils import RelayCache
    except ImportError:
        logging.warning("Could not import RelayCache. Ensure RelayCaching package is installed or path is correct.")

# Agent dataset support
try:
    from RelayCaching.prompt.prompt_set_registry import PromptSetRegistry
    from RelayCaching.tools.reader.readers import JSONLReader
    from datasets_self.gsm8k_dataset import universal_data_process
except ImportError:
    # Fallback: create empty placeholders if import fails
    PromptSetRegistry = None
    JSONLReader = None
    universal_data_process = None
    logging.warning("Could not import PromptSetRegistry/JSONLReader. Agent dataset support disabled.")

# Layer auto-selection (U-shape + L_det)
try:
    from experiments.observation.layer_selection import (
        auto_select_middle_layer_range,
        auto_select_middle_layer_range_from_kv,
        auto_select_detection_layer,
    )
except ImportError:
    try:
        # Fallback: relative path when running from different working directories
        import importlib.util
        _layer_sel_path = current_file_path.parents[2] / "experiments" / "observation" / "layer_selection.py"
        if _layer_sel_path.exists():
            _spec = importlib.util.spec_from_file_location("layer_selection", _layer_sel_path)
            _layer_sel_mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_layer_sel_mod)
            auto_select_middle_layer_range = _layer_sel_mod.auto_select_middle_layer_range
            auto_select_middle_layer_range_from_kv = _layer_sel_mod.auto_select_middle_layer_range_from_kv
            auto_select_detection_layer = _layer_sel_mod.auto_select_detection_layer
        else:
            auto_select_middle_layer_range = None
            auto_select_middle_layer_range_from_kv = None
            auto_select_detection_layer = None
            logging.warning("Could not import layer_selection. Auto layer range selection disabled.")
    except Exception:
        auto_select_middle_layer_range = None
        auto_select_middle_layer_range_from_kv = None
        auto_select_detection_layer = None
        logging.warning("Could not import layer_selection. Auto layer range selection disabled.")

try:
    from experiments.observation.plotting import plot_layer_profile_selection
except ImportError:
    try:
        _plot_path = current_file_path.parents[2] / "experiments" / "observation" / "plotting.py"
        if _plot_path.exists():
            import importlib.util
            _plot_spec = importlib.util.spec_from_file_location("observation_plotting", _plot_path)
            _plot_mod = importlib.util.module_from_spec(_plot_spec)
            _plot_spec.loader.exec_module(_plot_mod)
            plot_layer_profile_selection = _plot_mod.plot_layer_profile_selection
        else:
            plot_layer_profile_selection = None
    except Exception:
        plot_layer_profile_selection = None

# Circular import workaround
try:
    from RelayCaching.utils.profiler_utils import (
        load_model,
        load_dataset,
        normalize_question,
        build_prefix,
        build_suffix,
        ROPE_Module,
        enable_model_attention_prefill_context,
        extract_KV_from_outputs,
        extract_kv_with_tokens,
        kv_cache_apply_rope,
        kv_cache_inverse_rope,
        calculate_overall_similarity,
        FILTER_PHRASES,
        MAP_PROMPT_TEMPLATE,
        REDUCE_PROMPT_PREFIX_TEMPLATE,
        REDUCE_PROMPT_SUFFIX_TEMPLATE,
    )
except ImportError:
    # Fallback for running inside the utils directory directly
    from profiler_utils import (
        load_model,
        load_dataset,
        normalize_question,
        build_prefix,
        build_suffix,
        ROPE_Module,
        enable_model_attention_prefill_context,
        extract_KV_from_outputs,
        extract_kv_with_tokens,
        kv_cache_apply_rope,
        kv_cache_inverse_rope,
        calculate_overall_similarity,
        FILTER_PHRASES,
        MAP_PROMPT_TEMPLATE,
        REDUCE_PROMPT_PREFIX_TEMPLATE,
        REDUCE_PROMPT_SUFFIX_TEMPLATE,
    )



def compare_kv_data(data1, data2, atol=1e-5, rtol=1e-5):
    """
    Compare two KV Cache data structures for consistency.
    
    Args:
        data1, data2: Data returned by extract_kv_with_tokens
        atol: Absolute tolerance (for floating point precision)
        rtol: Relative tolerance
    """
    if len(data1) != len(data2):
        print(f"Layer count mismatch: {len(data1)} vs {len(data2)}")
        return False

    all_match = True

    for layer_idx, (layer_a, layer_b) in enumerate(zip(data1, data2)):
        if isinstance(layer_a, (tuple, list)) and len(layer_a) == 2:
            k1, v1 = layer_a
            k2, v2 = layer_b
        else:
            k1, v1 = layer_a, None
            k2, v2 = layer_b, None
        prefix_len =min(k1.shape[-2], k2.shape[-2])
        k1 = k1[:,:,:prefix_len,:]
        k2 = k2[:,:,:prefix_len,:]
        v1 = v1[:,:,:prefix_len,:] if v1 is not None else None
        v2 = v2[:,:,:prefix_len,:] if v2 is not None else None
        if not torch.allclose(k1, k2, atol=atol, rtol=rtol):
            diff = (k1 - k2).abs().max().item()
            print(f"Layer {layer_idx} [Key] mismatch! Max Diff: {diff}")
            all_match = False

        if v1 is not None:
            if not torch.allclose(v1, v2, atol=atol, rtol=rtol):
                diff = (v1 - v2).abs().max().item()
                print(f"Layer {layer_idx} [Value] mismatch! Max Diff: {diff}")
                all_match = False

    if all_match:
        print("  Two KV Cache results are identical")
        return True
    else:
        return False
        
class LayerRangeProfiler:
    """
    Profile intermediate layer ranges for KV cache reuse.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        
        

        # Extract core config
        self.st_idx = config["st_idx"]
        self.data_len = config["data_len"]
        self.ctx_num = config["ctx_num"]
        self.not_use_filter = config.get("not_use_filter", False)
        self.test_ctx_num = config.get("test_ctx_num", None)
        self.model_name = config["model_name"]
        
        # Agent dataset support parameters
        self.dataset_type = config.get("dataset_type", "wikimqa")  # "wikimqa", "gsm8k", "math", "humaneval"
        self.domain = config.get("domain", None)  # prompt domain, e.g. "gsm8k", "math"
        self.role = config.get("role", None)  # Agent role
        self.prompt_set = None  # Initialized in _load_profiling_data
        self.device_str = config["device"]
        self.num_layers = config.get("num_layers", 32)
        self.rec_layers = list(range(self.num_layers))

        

        # Define paths
        self.base_output_dir = Path(config["base_output_dir"])
        self.log_dir = self.base_output_dir / "logs"
        self.cache_dir = self.base_output_dir / "kv_caches"
        self.plot_dir = self.base_output_dir / "plots"
        self.logger = self._init_logger()
        # --- Internal state (populated during setup) ---
        self.device = None
        self.model = None
        self.tokenizer = None
        self.rope_module = None
        
        self.eval_dataset = None
        self.map_results = {}
        
        # Pre-computed caches and IDs
        self.prefix_cache: DynamicCache = None
        self.prefix_ids: torch.Tensor = None
        self.prefix_len: int = 0
        self.information_cache_list: List[DynamicCache] = []
        self.full_input_ids: torch.Tensor = None
        self.partial_len: int = 0
        self.suffix_len: int = 0

        # Layer selection: Decoding vs Prefill (observation-style)
        self.run_layer_sweep = config.get("run_layer_sweep", False)
        self.layer_lower_threshold = config.get("layer_lower_threshold", 0.99)
        self.layer_upper_sigma_multiplier = config.get("layer_upper_sigma_multiplier", 2.0)
        self.layer_upper_consecutive_count = config.get("layer_upper_consecutive_count", 2)
        self.layer_upper_min_samples = config.get("layer_upper_min_samples", 3)
        self.right_tail_layers = config.get("right_tail_layers", 5)
        self.layer_selection_metric = config.get("layer_selection_metric", "value_cosine")
        self.layer_override = config.get("layer_override")
        self._calibration_question: str = ""
        self._calibration_summary_token_ids: List[torch.Tensor] = []
        self._calibration_map_prompt_lens: List[int] = []
        self._token_deviations_cache: Dict[int, np.ndarray] = {}
        self._spearman_correlations_cache: Dict[int, float] = {}
        self._second_order_diffs_cache: Dict[int, float] = {}
        
        # Layer indices for legacy relay-vs-gold sweep only (--run-layer-sweep)
        self.sweep_values: Optional[List[int]] = config.get("sweep_values")

        self.default_params = {
            "det_layer": config.get("det_layer", 0),
            "end_layer": config.get("end_layer", self.num_layers - 1),
            "tau_dev": config.get("tau_dev", 0.0),
            "tau_inf": config.get("tau_inf", 0.0),
            "prefix_tokens": config.get("prefix_tokens", 0),
            "suffix_tokens": config.get("suffix_tokens", 0),
            "reuse_hidden_layer": config.get("reuse_hidden_layer", 0),
        }
        if self.run_layer_sweep and self.sweep_values is not None:
            self.logger.info(f"Profiler layer sweep values: {self.sweep_values}")

        # For analysis
        self.reference_cache: List[Dict] = None # Reference KV cache
        self.st_idx_data_key = str(self.st_idx) # For saving/loading pickle

        self.reuse_hidden_layer = config.get("reuse_hidden_layer", -1)
        self.store_hidden_layer = config.get("store_hidden_layer", -1)

        self.all_layer_hidden_states: Dict[int, List[torch.Tensor]] = {}
        self.reusable_hidden_states_list: List[torch.Tensor] = [] # Hidden states for RelayCache

    def _init_logger(self) -> logging.Logger:
        # Logger initialization
        logger = logging.getLogger(f"Profiler_st{self.st_idx}")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            return logger
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        log_file = self.base_output_dir / f"profiler_run_st{self.st_idx}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
    

    def _load_map_results_from_file(self):
        """
        Skip the Map phase and load pre-computed Map results directly from file.
        """
        map_path = Path(self.config["map_results_path"])
        self.logger.info(f"Loading Map results from file: {map_path}...")
        
        if not map_path.exists():
            self.logger.error(f"Map results file not found: {map_path}")
            self.logger.error("Please set 'run_map_phase': True to generate the file, or provide the correct path.")
            raise FileNotFoundError(f"Map results file not found: {map_path}")
        
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                self.map_results = json.load(f)
            self.logger.info("Pre-computed Map results loaded successfully.")
        except json.JSONDecodeError as e:
            self.logger.error(f"Cannot parse Map results file: {map_path}. File may be corrupted. {e}")
            raise

    def _run_map_phase(self):
        """
        Execute Map phase: generate summaries/information for each context.
        Supports multiple dataset types:
        - wikimqa: generate summary for each context
        - gsm8k/math/humaneval: use PromptSetRegistry to generate response
        
        Results are saved in self.map_results.
        """
        self.logger.info("="*30 + " Starting MAP Phase (in-memory) " + "="*30)
        
        # --- 1. Get required variables ---
        start_idx = self.config["st_idx"]
        data_len = self.config["data_len"]
        model_name = self.config["model_name"]
        max_ctx_num = self.config["ctx_num"]
        
        # --- 2. Ensure model and data are loaded ---
        if self.model is None or self.tokenizer is None:
            self.logger.error("Map phase failed: model not loaded.")
            raise RuntimeError("Model not loaded before running map phase.")
        if self.eval_dataset is None:
            self.logger.error("Map phase failed: eval dataset not loaded.")
            raise RuntimeError("Eval dataset not loaded before running map phase.")

        # --- 3. Process data loop ---
        temp_all_results = {}
        for data_idx in range(start_idx, start_idx + data_len):
            self.logger.info(f"\n===== Processing Map data {data_idx} =====")
            try:
                qa_pair = self.eval_dataset[data_idx]
            except IndexError:
                self.logger.error(f"Data index {data_idx} out of dataset range")
                continue

            information_list = []
            inputs_len_list = []
            
            # Process based on dataset type
            if self.dataset_type == "wikimqa":
                # WikiMQA mode: generate summary for each context
                question = normalize_question(qa_pair["question"])
                valid_ctx_count = 0

                for ctx_idx, ctx in enumerate(qa_pair["ctxs"]):
                    context = ctx["text"]
                    map_prompt = MAP_PROMPT_TEMPLATE.format(question=question, context=context)
                    
                    # Build full prompt
                    map_prompt_full = build_prefix(model_name, map_prompt)
                    map_prompt_full = build_suffix(model_name, map_prompt_full)
                    
                    inputs = self.tokenizer(map_prompt_full, return_tensors="pt", add_special_tokens=False).to(self.device)
                    inputs_len = inputs["input_ids"].shape[1]
                    
                    with torch.no_grad():
                        full_outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            use_cache=True,
                            do_sample=False,
                            return_dict_in_generate=True,
                        )
                    
                    # Save sequence with prompt, removing last token
                    generated_ids = full_outputs["sequences"][:, :-1]
                    summary_text = self.tokenizer.decode(
                        generated_ids[0, inputs_len:], skip_special_tokens=True,
                    )

                    if not self.not_use_filter and any(
                        phrase in summary_text.lower() for phrase in FILTER_PHRASES
                    ):
                        self.logger.warning(
                            f"  Ctx {ctx_idx}: skipped in MAP (filter phrase). "
                            f"Preview: '{summary_text[:80]}...'"
                        )
                        continue

                    valid_ctx_count += 1
                    self.logger.info(
                        f"  Ctx {ctx_idx}: input {inputs_len} tokens, "
                        f"generated {generated_ids.shape[1] - inputs_len} tokens "
                        f"(Valid #{valid_ctx_count})."
                    )
                    information_list.append(generated_ids.cpu().numpy().tolist()[0])
                    inputs_len_list.append(inputs_len)

                    if valid_ctx_count >= max_ctx_num:
                        self.logger.info(
                            f"  Reached ctx_num={max_ctx_num}, stopping MAP for data {data_idx}."
                        )
                        break

                if len(information_list) == 0:
                    self.logger.warning(
                        f"  Data {data_idx}: no valid summaries after MAP filter "
                        f"(ctx_num={max_ctx_num})."
                    )
                    
            else:
                # Agent mode (gsm8k/math/humaneval): use PromptSetRegistry to generate response
                # Extract question/task: prefer 'task', then 'question', then 'prompt'
                question = None
                for key in ["task", "question", "prompt", "problem"]:
                    if key in qa_pair and qa_pair[key]:
                        question = qa_pair[key]
                        break
                
                if question is None:
                    self.logger.error(f"Data format error: cannot extract question, available fields: {list(qa_pair.keys())}")
                    continue
                
                self.logger.info(f"   Question extracted (field: {key}), length: {len(question)}")
                
                # Build MAP prompt
                if self.prompt_set:
                    # Use formal task prompt system
                    system_prompt = self.prompt_set.get_constraint(self.role)
                    
                    # Different domains use different get_answer_prompt parameters
                    if self.dataset_type == "humaneval":
                        user_prompt = self.prompt_set.get_answer_prompt(question=question)
                    else:
                        user_prompt = self.prompt_set.get_answer_prompt(question=question, role=self.role)
                    
                    # Combine system + user prompt
                    map_prompt = f"{system_prompt}\n\n{user_prompt}"
                    self.logger.info(f"   System prompt: {system_prompt[:100]}...")
                else:
                    # Fallback: use simple prompt
                    if self.dataset_type == "humaneval":
                        map_prompt = f"{question}\nWrite your full implementation in a Python code block.\n"
                    else:
                        map_prompt = f"Q: {question}\nA: Let's think step by step.\n"
                    self.logger.info(f"   Using simple prompt (no PromptSet)")

                map_prompt_full = build_prefix(model_name, map_prompt)
                map_prompt_full = build_suffix(model_name, map_prompt_full)

                inputs = self.tokenizer(map_prompt_full, return_tensors="pt", add_special_tokens=False).to(self.device)
                inputs_len = inputs["input_ids"].shape[1]
                
                # Adjust max tokens based on dataset type
                max_new_tokens = 512
                if self.dataset_type == "humaneval":
                    max_new_tokens = 1024  # Code generation may need more tokens
                
                with torch.no_grad():
                    full_outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        do_sample=False,
                        return_dict_in_generate=True,
                    )
                
                # Save sequence with prompt, removing last token
                generated_ids = full_outputs["sequences"][:, :-1]
                
                self.logger.info(
                    f"  Agent Response: input {inputs_len} tokens, "
                    f"generated {generated_ids.shape[1] - inputs_len} tokens."
                )
                information_list.append(generated_ids.cpu().numpy().tolist()[0])
                inputs_len_list.append(inputs_len)

            data_result = {
                "summaries": information_list,
                "inputs_lens": inputs_len_list
            }
            temp_all_results[str(data_idx)] = data_result
        
        # --- 4. Save to class attribute ---
        self.map_results = temp_all_results
        self.logger.info("Map results generated and saved in memory.")
        
        # --- 5. [Optional] Save to file for debugging ---
        map_results_path = Path(self.config["map_results_path"])
        map_results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(map_results_path, 'w') as f:
            json.dump(self.map_results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Map results copy saved to {map_results_path} for debugging.")
        
        self.logger.info("="*30 + " MAP Phase Complete " + "="*30)

    def _setup_directories(self):
        # Directory setup
        self.logger.info("Setting up output directories...")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "low").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "high").mkdir(parents=True, exist_ok=True)

    def _load_model_and_tokenizer(self):
        """
        Load model, tokenizer and RoPE module at class initialization (runs once).
        Uses imported 'load_model' and 'ROPE_Module'.
        """
        self.logger.info(f"Loading model: {self.model_name} to {self.device_str}...")
        self.device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")
        
        # Use imported function from profiler_utils.py
        self.model, self.tokenizer = load_model(
            model_name=self.model_name,
            quantization=None,
            device_map=self.device_str,
            torch_dtype=torch.bfloat16,
        )
        # Use imported function
        enable_model_attention_prefill_context(self.model_name)
        # Use imported class
        self.rope_module = ROPE_Module(model_name=self.model_name, device=self.device)
        
        self.full_num_layers = self.model.config.num_hidden_layers
        if self.num_layers == -1:
            self.num_layers = self.full_num_layers
            self.rec_layers = list(range(self.full_num_layers))
        self.logger.info(f"Model loaded. Detected {self.full_num_layers} layers.")

    def _load_profiling_data(self):
        """
        Load profiling data, supporting multiple dataset types:
        - wikimqa: QA with contexts (original format)
        - gsm8k/math: math problems
        - humaneval: code generation
        """
        self.logger.info("Loading profiling data...")
        dataset_path = self.config["eval_dataset_path"]

        if self.dataset_type == "wikimqa":
            self.eval_dataset = load_dataset(dataset_path)
            self.logger.info(f"   Dataset type: wikimqa (QA with contexts)")

        elif self.dataset_type == "humaneval":
            if JSONLReader is None:
                raise ImportError("JSONLReader not available. Cannot load humaneval dataset.")
            if dataset_path.endswith(".jsonl"):
                dataset_raw = JSONLReader.parse_file(dataset_path)
            else:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset_raw = json.load(f)
            self.eval_dataset = [{"task": item["prompt"], "test": item.get("test", "")} for item in dataset_raw]
            self.logger.info(f"   Dataset type: humaneval (code generation)")

        elif self.dataset_type in ["gsm8k", "math"]:
            if JSONLReader is None or universal_data_process is None:
                raise ImportError("JSONLReader/universal_data_process not available. Cannot load math dataset.")
            if dataset_path.endswith(".jsonl"):
                dataset_raw = JSONLReader.parse_file(dataset_path)
            else:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    dataset_raw = json.load(f)
            self.eval_dataset = universal_data_process(dataset_raw)
            self.logger.info(f"   Dataset type: {self.dataset_type} (math problems)")

        elif self.dataset_type == "mmlu":
            from datasets_self.mmlu_dataset import load_mmlu_observation_records
            self.eval_dataset = load_mmlu_observation_records(dataset_path)
            self.logger.info("   Dataset type: mmlu (multiple-choice, CSV)")

        else:
            try:
                self.eval_dataset = load_dataset(dataset_path)
                self.logger.info(f"   Dataset type: {self.dataset_type} (generic)")
            except Exception:
                if JSONLReader is not None:
                    dataset_raw = JSONLReader.parse_file(dataset_path)
                    if universal_data_process is not None:
                        self.eval_dataset = universal_data_process(dataset_raw)
                    else:
                        self.eval_dataset = dataset_raw
                else:
                    raise

        if PromptSetRegistry is not None:
            if self.domain:
                self.prompt_set = PromptSetRegistry.get(self.domain)
            elif self.dataset_type in ["gsm8k", "math", "humaneval", "mmlu"]:
                self.domain = self.dataset_type
                self.prompt_set = PromptSetRegistry.get(self.domain)
                self.logger.info(f"   Auto-detected domain: {self.domain}")

            if self.prompt_set:
                if self.role is None:
                    self.role = self.prompt_set.get_role()
                self.logger.info(f"   Using PromptSet: {self.domain}, Role: {self.role}")
            elif self.domain:
                self.logger.warning(f"   Warning: domain '{self.domain}' specified but PromptSet not found")
                self.domain = None

        self.logger.info(f"Data loading complete. Total {len(self.eval_dataset)} records.")

    def _merge_layer_similarity_lists(self, per_sample_sims: List[List[Dict]]) -> List[Dict]:
        """Average per-layer metrics across calibration samples, then run auto_select once."""
        if not per_sample_sims:
            return []
        merged = []
        for layer_idx in range(len(per_sample_sims[0])):
            merged.append({
                key: float(np.mean([sample[layer_idx][key] for sample in per_sample_sims]))
                for key in per_sample_sims[0][layer_idx].keys()
            })
        return merged

    def _merge_token_deviation_dicts(
        self, per_sample_devs: List[Dict[int, np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        """Average token-wise deviations per layer (same token count required)."""
        if not per_sample_devs:
            return {}
        merged: Dict[int, np.ndarray] = {}
        all_layers = set()
        for devs in per_sample_devs:
            all_layers.update(devs.keys())
        for layer_idx in sorted(all_layers):
            arrays = [devs[layer_idx] for devs in per_sample_devs if layer_idx in devs]
            if not arrays:
                continue
            min_len = min(a.shape[0] for a in arrays)
            if min_len == 0:
                continue
            stacked = np.stack([a[:min_len] for a in arrays], axis=0)
            merged[layer_idx] = np.mean(stacked, axis=0)
        return merged

    def _apply_rope_to_layer_kv_list(
        self, kv_list: List[Dict], position_ids: torch.Tensor,
    ) -> DynamicCache:
        cache = DynamicCache()
        for layer_kv in kv_list:
            cache.update(
                layer_kv["key"].unsqueeze(0),
                layer_kv["value"].unsqueeze(0),
                layer_idx=len(cache.key_cache),
            )
        return kv_cache_apply_rope(cache, position_ids, self.rope_module)

    def _collect_prefill_kv_for_layer_select(
        self,
    ) -> Tuple[List[List[Dict]], List[int], List[int]]:
        """Prefill summaries in reduce prompt (observation Step 3)."""
        question = self._calibration_question
        summary_token_ids = self._calibration_summary_token_ids
        map_prompt_lens = self._calibration_map_prompt_lens

        info_start_positions: List[int] = []
        info_lengths: List[int] = []
        all_info_ids_list: List[torch.Tensor] = []

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
            for agent_idx in range(len(summary_token_ids)):
                agent_role = self.role if self.role else "Agent"
                agent_header = f"Agent {agent_idx + 1}, role is {agent_role}, output is:\n\n"
                header_ids = self.tokenizer(
                    agent_header, return_tensors="pt", add_special_tokens=False,
                ).input_ids.to(self.device)
                spatial_segments_ids.append(header_ids)
                current_pos += header_ids.shape[1]

                summary_start = current_pos
                if agent_idx < len(map_prompt_lens):
                    summary_ids = summary_token_ids[agent_idx][map_prompt_lens[agent_idx] :].unsqueeze(0).to(self.device)
                else:
                    summary_ids = summary_token_ids[agent_idx].unsqueeze(0).to(self.device)
                spatial_segments_ids.append(summary_ids)
                summary_len = summary_ids.shape[1]
                current_pos += summary_len

                info_start_positions.append(summary_start)
                info_lengths.append(summary_len)
                all_info_ids_list.append(summary_ids)

                suffix_ids = self.tokenizer(
                    "\n\n", return_tensors="pt", add_special_tokens=False,
                ).input_ids.to(self.device)
                spatial_segments_ids.append(suffix_ids)
                current_pos += suffix_ids.shape[1]

            after_spatial_ids = self.tokenizer(
                "\n", return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(self.device)

            if self.domain in ["humaneval", "mbpp", "ds1000", "humaneval_plus"]:
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
            if self.dataset_type == "wikimqa":
                reduce_prompt_prefix = REDUCE_PROMPT_PREFIX_TEMPLATE.format(question=question)
                reduce_prompt_suffix = REDUCE_PROMPT_SUFFIX_TEMPLATE.format(question=question)
            elif self.dataset_type == "humaneval":
                reduce_prompt_prefix = (
                    f"Based on the function signature below and the provided implementations, "
                    f"write the final implementation.\n{question}\nImplementations:\n\n"
                )
                reduce_prompt_suffix = "Write your final implementation.\n```python\n"
            elif self.dataset_type in ["gsm8k", "math"]:
                reduce_prompt_prefix = f"Answer the question based on the given solutions.\nQuestion:{question}\nSolutions:\n\n"
                reduce_prompt_suffix = f"Answer the question within 5 words.\nQuestion:{question}\nAnswer:"
            else:
                reduce_prompt_prefix = f"Answer the question based on the given informations.\nQuestion:{question}\nInfos:\n\n"
                reduce_prompt_suffix = f"Answer the question within 5 words.\nQuestion:{question}\nAnswer:"

            reduce_prompt_prefix = build_prefix(self.model_name, reduce_prompt_prefix)
            reduce_prefix_ids = self.tokenizer(
                reduce_prompt_prefix, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(self.device)
            current_pos = reduce_prefix_ids.shape[1]

            for summary_ids_full, map_prompt_len in zip(summary_token_ids, map_prompt_lens):
                summary_only_ids = summary_ids_full[map_prompt_len:].unsqueeze(0).to(self.device)
                all_info_ids_list.append(summary_only_ids)
                info_start_positions.append(current_pos)
                info_lengths.append(summary_only_ids.shape[1])
                current_pos += summary_only_ids.shape[1]

            reduce_prompt_suffix = build_suffix(self.model_name, reduce_prompt_suffix)
            reduce_suffix_ids = self.tokenizer(
                reduce_prompt_suffix, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(self.device)
            full_reduce_ids = torch.cat([reduce_prefix_ids] + all_info_ids_list + [reduce_suffix_ids], dim=1)

        with torch.no_grad():
            reduce_outputs = self.model(input_ids=full_reduce_ids, use_cache=True, return_dict=True)

        prefill_kv_list: List[List[Dict]] = []
        for start_pos, summary_len in zip(info_start_positions, info_lengths):
            temp_prefill_cache = extract_KV_from_outputs(
                reduce_outputs.past_key_values, start_pos, summary_len,
            )
            position_ids = torch.tensor(
                list(range(start_pos, start_pos + summary_len)),
                device=self.device,
            ).unsqueeze(0)
            prefill_kv_no_rope = kv_cache_inverse_rope(temp_prefill_cache, position_ids, self.rope_module)
            prefill_kv = []
            for layer_idx in range(self.num_layers):
                prefill_kv.append({
                    "key": prefill_kv_no_rope.key_cache[layer_idx].squeeze(0),
                    "value": prefill_kv_no_rope.value_cache[layer_idx].squeeze(0),
                })
            prefill_kv_list.append(prefill_kv)

        return prefill_kv_list, info_start_positions, info_lengths

    def _build_decoding_kv_list_for_layer_select(
        self, info_start_positions: List[int], info_lengths: List[int],
    ) -> List[List[Dict]]:
        """Decoding KV per agent with reduce-position RoPE (observation Step 9)."""
        decoding_kv_list: List[List[Dict]] = []
        for ctx_idx, info_cache in enumerate(self.information_cache_list):
            start_pos = info_start_positions[ctx_idx]
            summary_len = info_lengths[ctx_idx]
            decoding_kv = []
            for layer_idx in range(self.num_layers):
                decoding_kv.append({
                    "key": info_cache.key_cache[layer_idx].squeeze(0),
                    "value": info_cache.value_cache[layer_idx].squeeze(0),
                })
            reduce_position_ids = torch.tensor(
                list(range(start_pos, start_pos + summary_len)),
                device=self.device,
            ).unsqueeze(0)
            decode_cache = self._apply_rope_to_layer_kv_list(decoding_kv, reduce_position_ids)
            for layer_idx in range(self.num_layers):
                decoding_kv[layer_idx]["key"] = decode_cache.key_cache[layer_idx].squeeze(0)
                decoding_kv[layer_idx]["value"] = decode_cache.value_cache[layer_idx].squeeze(0)
            decoding_kv_list.append(decoding_kv)
        return decoding_kv_list

    def _compute_layer_similarities_observation(self, data_idx: int) -> List[Dict]:
        """Per-layer Decoding vs Prefill (Paper Fig.2 / original observation Step 5.5)."""
        self.logger.info(f"[Data {data_idx}] Observation-style layer profile (Decoding vs Prefill)...")
        prefill_kv_list, info_start_positions, info_lengths = self._collect_prefill_kv_for_layer_select()
        decoding_kv_list = self._build_decoding_kv_list_for_layer_select(
            info_start_positions, info_lengths,
        )

        for ctx_idx, (start_pos, summary_len) in enumerate(zip(info_start_positions, info_lengths)):
            reduce_position_ids = torch.tensor(
                list(range(start_pos, start_pos + summary_len)),
                device=self.device,
            ).unsqueeze(0)
            prefill_cache = self._apply_rope_to_layer_kv_list(
                prefill_kv_list[ctx_idx], reduce_position_ids,
            )
            for layer_idx in range(self.num_layers):
                prefill_kv_list[ctx_idx][layer_idx]["key"] = prefill_cache.key_cache[layer_idx].squeeze(0)
                prefill_kv_list[ctx_idx][layer_idx]["value"] = prefill_cache.value_cache[layer_idx].squeeze(0)

        if auto_select_middle_layer_range_from_kv is not None:
            auto_select_middle_layer_range_from_kv(
                decoding_kv_list=decoding_kv_list,
                prefill_kv_list=prefill_kv_list,
                num_layers=self.num_layers,
                lower_threshold=self.layer_lower_threshold,
                upper_sigma_multiplier=self.layer_upper_sigma_multiplier,
                upper_consecutive_count=self.layer_upper_consecutive_count,
                upper_min_samples=self.layer_upper_min_samples,
                right_tail_layers=self.right_tail_layers,
                metric=self.layer_selection_metric,
                verbose=True,
            )

        layer_similarities = []
        for layer_idx in range(self.num_layers):
            key_cos, val_cos = [], []
            for dec_kv, pre_kv in zip(decoding_kv_list, prefill_kv_list):
                dec_k, dec_v = dec_kv[layer_idx]["key"], dec_kv[layer_idx]["value"]
                pre_k, pre_v = pre_kv[layer_idx]["key"], pre_kv[layer_idx]["value"]
                if dec_k.shape != pre_k.shape or dec_k.shape[0] == 0:
                    continue
                key_cos.append(F.cosine_similarity(dec_k, pre_k, dim=1).mean().item())
                val_cos.append(F.cosine_similarity(dec_v, pre_v, dim=1).mean().item())
            layer_similarities.append({
                "key_cosine": float(np.mean(key_cos)) if key_cos else 0.0,
                "value_cosine": float(np.mean(val_cos)) if val_cos else 0.0,
                "key_norm": 0.0,
                "value_norm": 0.0,
            })
        return layer_similarities

    def _collect_token_deviations_dec_prefill(self, ctx_idx: int = 0) -> None:
        """Token-wise deviation between Decoding and Prefill for L_det (Paper Section 4.2)."""
        prefill_kv_list, info_start_positions, info_lengths = self._collect_prefill_kv_for_layer_select()
        decoding_kv_list = self._build_decoding_kv_list_for_layer_select(
            info_start_positions, info_lengths,
        )
        if ctx_idx >= len(decoding_kv_list):
            return

        start_pos = info_start_positions[ctx_idx]
        summary_len = info_lengths[ctx_idx]
        reduce_position_ids = torch.tensor(
            list(range(start_pos, start_pos + summary_len)),
            device=self.device,
        ).unsqueeze(0)

        prefill_rope = self._apply_rope_to_layer_kv_list(
            prefill_kv_list[ctx_idx], reduce_position_ids,
        )
        decode_rope = self._apply_rope_to_layer_kv_list(
            decoding_kv_list[ctx_idx], reduce_position_ids,
        )

        token_deviations: Dict[int, np.ndarray] = {}
        for layer_idx in range(self.num_layers):
            dec_v = decode_rope.value_cache[layer_idx]
            pre_v = prefill_rope.value_cache[layer_idx]
            if dec_v.dim() == 4:
                dec_flat = dec_v[0].permute(1, 0, 2).reshape(dec_v.shape[2], -1)
                pre_flat = pre_v[0].permute(1, 0, 2).reshape(pre_v.shape[2], -1)
            elif dec_v.dim() == 3:
                dec_flat = dec_v[0]
                pre_flat = pre_v[0]
            else:
                continue
            min_tokens = min(dec_flat.shape[0], pre_flat.shape[0])
            if min_tokens == 0:
                continue
            cos_sim = F.cosine_similarity(dec_flat[:min_tokens], pre_flat[:min_tokens], dim=-1)
            token_deviations[layer_idx] = (1.0 - cos_sim).float().cpu().numpy()

        if token_deviations:
            self._token_deviations_cache = token_deviations
            self.logger.info(
                f"  Token deviations (Dec vs Prefill) collected for {len(token_deviations)} layers."
            )

    def _run_gold_baseline(self) -> List[Dict]:
        """
        Run standard Full Prefill baseline (without RelayCache) and extract Suffix Cache.
        """
        self.logger.info("--> Running GOLD BASELINE (Full Prefill).")

        gold_cache = DynamicCache()
        for l in range(self.full_num_layers):
            gold_cache.key_cache.append(self.final_cache_with_rope.key_cache[l][:,:,:self.prefix_len,:].clone())
            gold_cache.value_cache.append(self.final_cache_with_rope.value_cache[l][:,:,:self.prefix_len,:].clone())

        with torch.no_grad():
            full_outputs1 = self.model.generate(
                self.full_input_ids,
                max_new_tokens=self.config.get("max_new_tokens", 2),
                use_cache=True,
                do_sample=False,
                past_key_values=gold_cache,
                return_dict_in_generate=True,
                output_attentions=False,
            )

        gold_cache_data = extract_kv_with_tokens(
            full_outputs1,
            self.partial_len,
            self.suffix_len
        )
        self.logger.info("  GOLD BASELINE cache extraction complete.")
        return gold_cache_data

    def _collect_token_deviations(self):
        """
        Collect token-wise deviation data for determining L_det.

        Compute token-wise value cosine similarity deviation between
        decoding KV and gold baseline (prefill KV) for each layer.

        Results are stored in self._token_deviations_cache as:
        {layer_idx: np.array of deviations (1 - cosine_sim) per token}
        """
        if self.reference_cache is None or not self.information_cache_list:
            self.logger.warning("Cannot collect token deviations: missing reference or information cache.")
            return
        
        self.logger.info("--> Collecting token-wise deviations for L_det determination...")
        
        token_deviations = {}

        if len(self.information_cache_list) == 0:
            return

        decoding_kv = self.information_cache_list[0]

        for layer_idx in range(self.num_layers):
            try:
                ref_layer = self.reference_cache[layer_idx]
                if isinstance(ref_layer, (tuple, list)) and len(ref_layer) == 2:
                    ref_v = ref_layer[1]
                elif isinstance(ref_layer, dict):
                    ref_v = ref_layer.get('value', ref_layer.get('v'))
                else:
                    continue

                if hasattr(decoding_kv, 'value_cache'):
                    dec_v = decoding_kv.value_cache[layer_idx]
                elif isinstance(decoding_kv, (list, tuple)):
                    if isinstance(decoding_kv[layer_idx], (tuple, list)):
                        dec_v = decoding_kv[layer_idx][1]
                    elif isinstance(decoding_kv[layer_idx], dict):
                        dec_v = decoding_kv[layer_idx].get('value', decoding_kv[layer_idx].get('v'))
                    else:
                        continue
                else:
                    continue

                if ref_v is None or dec_v is None:
                    continue

                if ref_v.dim() == 4:
                    ref_v_flat = ref_v[0].permute(1, 0, 2).reshape(ref_v.shape[2], -1)
                    dec_v_flat = dec_v[0].permute(1, 0, 2).reshape(dec_v.shape[2], -1)
                elif ref_v.dim() == 3:
                    ref_v_flat = ref_v[0]
                    dec_v_flat = dec_v[0]
                else:
                    continue

                min_tokens = min(ref_v_flat.shape[0], dec_v_flat.shape[0])
                if min_tokens == 0:
                    continue

                ref_v_flat = ref_v_flat[:min_tokens]
                dec_v_flat = dec_v_flat[:min_tokens]

                cos_sim = F.cosine_similarity(ref_v_flat, dec_v_flat, dim=-1)
                token_deviations[layer_idx] = (1.0 - cos_sim).float().cpu().numpy()
                
            except Exception as e:
                self.logger.debug(f"  Layer {layer_idx}: skipped ({e})")
                continue
        
        if token_deviations:
            self._token_deviations_cache = token_deviations
            self.logger.info(f"  Token deviations collected for {len(token_deviations)} layers.")
        else:
            self.logger.warning("  No token deviations could be collected.")

    def _run_no_repair_baseline(self, mode: str = "high") -> Tuple[List[Dict], Dict[str, float]]:
        """
        Run no-repair baseline for computing relative improvement ratio.

        The no-repair baseline uses the same profile settings as the experiment mode,
        but sets det_layer = num_layers so no layer is actually repaired.

        Args:
            mode: "high" for Upper Bound mode, "low" for Lower Bound mode

        Returns:
            (no_repair_cache_data, no_repair_similarity)
        """
        self.logger.info(f"--> Running NO REPAIR BASELINE ({mode} mode, no layer repaired).")

        profile_lower = (mode == "low")
        profile_upper = (mode == "high")

        current_layer_hidden_states = self.all_layer_hidden_states[0]

        no_repair_cache = RelayCache(
            self.final_cache_with_rope,
            recompute_indices = self.recompute_indices,
            blank_absolute_indices = [],
            blank_relative_indices = [],
            relay_absolute_indices = self.recompute_indices,
            relay_relative_indices = list(range(len(self.recompute_indices))),

            recompute_label = True,
            det_layer = self.full_num_layers + 1 if profile_lower else -1,
            end_layer = self.full_num_layers,

            tau_dev = 0.0,
            tau_inf = self.default_params.get("tau_inf", 0.0),
            prefix_tokens = self.default_params.get("prefix_tokens", 0),
            suffix_tokens = self.default_params.get("suffix_tokens", 0),

            reuse_hidden_layer = 0,
            store_hidden_layer = None,

            execution_mode = "relaycaching",
            additional_indices = self.addition_indices_list,
            relay_chunk_indices = self.ctx_indices_list,
            profile_lower_layer = profile_lower,
            profile_upper_layer = profile_upper,
            reusable_hidden_states_list = current_layer_hidden_states,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                self.full_input_ids,
                max_new_tokens=self.config.get("max_new_tokens", 2),
                use_cache=True,
                do_sample=False,
                past_key_values=no_repair_cache,
                return_dict_in_generate=True,
                output_attentions=False, # temporarily disabled for performance
            )

        no_repair_cache_data = extract_kv_with_tokens(
            outputs,
            self.partial_len,
            self.suffix_len
        )

        no_repair_similarity = calculate_overall_similarity(self.reference_cache, no_repair_cache_data)
        self.logger.info(f"  NO REPAIR BASELINE ({mode}) similarity: key_cos={no_repair_similarity.get('key_cosine_similarity', 'N/A'):.4f}, "
                        f"value_cos={no_repair_similarity.get('value_cosine_similarity', 'N/A'):.4f}")
        
        return no_repair_cache_data, no_repair_similarity

    def _aggregate_results(self, all_results: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """
        Compute average results across multiple data samples.
        """
        if not all_results:
            return {}

        keys = ["key_cosine", "value_cosine", "key_norm", "value_norm"]
        avg_results = {k: [] for k in keys}

        for k in keys:
            stacked_scores = []
            for res in all_results:
                if k in res:
                    stacked_scores.append(res[k])

            if stacked_scores:
                stacked_scores = np.array(stacked_scores)
                avg_scores = np.nanmean(stacked_scores, axis=0).tolist()
                avg_results[k] = avg_scores

        return avg_results
    
    def _prepare_all_caches(self, data_idx: int):
        """
        Prepare caches for the specified data_idx. Clear state from previous data before each call.
        """
        self.logger.info(f"Preparing all precomputed caches for data_idx={data_idx}...")
        self.reference_cache = None
        self.all_layer_hidden_states = {}
        self.information_cache_list = []
        self.recompute_index_list = []
        self._calibration_summary_token_ids = []
        self._calibration_map_prompt_lens = []

        try:
            qa_pair = self.eval_dataset[data_idx]
            map_data = self.map_results[str(data_idx)]
        except (IndexError, KeyError) as e:
            self.logger.error(f"Unable to get data or Map results for data_idx={data_idx}: {e}")
            raise e

        self.information_cache_list = []
        information_ids_list = []

        contexts_to_process = map_data["summaries"]
        inputs_lens_to_process = map_data["inputs_lens"]

        real_idx = 0
        self.recompute_index_list = []
        for idx, (full_info_ids, inputs_len) in enumerate(zip(contexts_to_process, inputs_lens_to_process)):
            # Filter + ctx_num cap are applied in _run_map_phase for newly generated map
            # results; keep this block for legacy map JSON and not_use_filter=False safety.
            if self.not_use_filter == False:
                assert(self.test_ctx_num == None)
                summary_text = self.tokenizer.decode(full_info_ids[inputs_len:], skip_special_tokens=True)
                summary_text_lower = summary_text.lower()
                if any(phrase in summary_text_lower for phrase in FILTER_PHRASES):
                    self.logger.warning(
                        f"Data {self.st_idx}, Ctx {idx}: skipped due to containing filtered phrase. Content: '{summary_text[:]}...'"
                    )
                    continue
            if real_idx >= self.ctx_num:
                self.logger.info(
                    "Maximum context count reached, stopping further processing "
                    f"(real_idx={real_idx}, ctx_num={self.ctx_num})."
                )
                break
            real_idx += 1
            self.logger.info(f"  Processing context {idx+1}/{len(contexts_to_process)}...")
            if self.config.get("is_prefix", True):
                full_info_ids_tensor = torch.tensor(full_info_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            else:
                full_info_ids_tensor = torch.tensor(full_info_ids[inputs_len:], dtype=torch.long, device=self.device).unsqueeze(0)

            prefix_ids = torch.tensor([], dtype=torch.long, device=self.device).unsqueeze(0)
            suffix_ids = torch.tensor([], dtype=torch.long, device=self.device).unsqueeze(0)

            if self.config.get("is_prefix", True):
                full_info_ids = torch.cat([full_info_ids_tensor[0][:inputs_len], prefix_ids[0][:], full_info_ids_tensor[0][inputs_len:], suffix_ids[0][:]], dim=0)
            else:
                full_info_ids = torch.cat([prefix_ids[0][:], full_info_ids_tensor[0][:], suffix_ids[0][:]], dim=0)

            full_info_ids = full_info_ids.unsqueeze(0)
            hidden_cache = DynamicCache()
            with torch.no_grad():
                full_outputs = self.model(
                    input_ids=full_info_ids,
                    use_cache=True,
                    past_key_values=hidden_cache,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            if self.config.get("is_prefix", True):
                generated_ids = full_info_ids[:, inputs_len:]
                generated_len = generated_ids.shape[1]
                pos_start = inputs_len
            else:
                generated_ids = full_info_ids
                generated_len = generated_ids.shape[1]
                pos_start = 0
            
            information_ids_list.append(generated_ids)
            for l_idx in range(self.full_num_layers):
                if l_idx not in self.all_layer_hidden_states:
                    self.all_layer_hidden_states[l_idx] = []
                
                hidden_states_full_tensor = full_outputs.hidden_states[l_idx]

                reusable_states_segment = hidden_states_full_tensor[
                    :,
                    pos_start : pos_start + generated_len,
                    :
                ].clone().detach().to(self.device)

                self.all_layer_hidden_states[l_idx].append(reusable_states_segment)
            if 'attentions' in full_outputs:
                temp_recompute_indices = list(range(generated_len))
                self.recompute_index_list.append(temp_recompute_indices)
            temp_decode_cache = extract_KV_from_outputs(full_outputs.past_key_values, pos_start, generated_len)
            position_ids = torch.tensor(list(range(pos_start, pos_start + generated_len)), device=self.device).unsqueeze(0)

            temp_kv_rope_removed = kv_cache_inverse_rope(temp_decode_cache, position_ids, self.rope_module)
            self.information_cache_list.append(temp_kv_rope_removed)
            self._calibration_summary_token_ids.append(full_info_ids[0].clone())
            self._calibration_map_prompt_lens.append(int(inputs_len))

        if len(self.information_cache_list) == 0:
            self.logger.warning(f"  Data {data_idx} skipped: all contexts filtered or no valid contexts.")
            return False

        self.logger.info("  Context cache computation complete.")

        if self.dataset_type == "wikimqa":
            question = normalize_question(qa_pair["question"])
        else:
            question = None
            for key in ["task", "question", "prompt", "problem"]:
                if key in qa_pair and qa_pair[key]:
                    question = qa_pair[key]
                    break

            if question is None:
                self.logger.warning(f"  Unable to extract question from qa_pair! Available fields: {list(qa_pair.keys())}")
                question = ""
            else:
                self.logger.info(f"   Question extracted successfully (using field: {key}), length: {len(question)}")

        self._calibration_question = question

        if self.dataset_type == "wikimqa":
            reduce_prompt_prefix = REDUCE_PROMPT_PREFIX_TEMPLATE.format(question=question)
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
            self.logger.info(f"   Using Agent Decision Prompt (domain: {self.domain})")
        else:
            reduce_prompt_prefix = f"Based on the following information, answer the question:\nQuestion: {question}\n\nInformation:\n"

        reduce_prompt_prefix = build_prefix(self.model_name, reduce_prompt_prefix)
        self.prefix_ids = self.tokenizer(
            reduce_prompt_prefix, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        self.prefix_len = self.prefix_ids.shape[1]
        
        self.prefix_cache_with_rope = DynamicCache()
        with torch.no_grad():
            outputs = self.model(
                self.prefix_ids,
                past_key_values=self.prefix_cache_with_rope,
                return_dict=True,
                use_cache=True,
            )
        cache_copy = DynamicCache()
        cache_copy.key_cache = [k.clone() for k in self.prefix_cache_with_rope.key_cache]
        cache_copy.value_cache = [v.clone() for v in self.prefix_cache_with_rope.value_cache]

        if hasattr(self.prefix_cache_with_rope, "_seen_tokens"):
            cache_copy._seen_tokens = self.prefix_cache_with_rope._seen_tokens
        position_ids = torch.tensor(list(range(self.prefix_len)), device=self.device).unsqueeze(0)
        self.prefix_cache = kv_cache_inverse_rope(cache_copy, position_ids, self.rope_module)
        self.logger.info("  Reduce prefix cache computation complete.")

        if self.dataset_type == "wikimqa":
            reduce_prompt_suffix = REDUCE_PROMPT_SUFFIX_TEMPLATE.format(question=question)
        elif self.prompt_set:
            if self.domain in ["humaneval", "mbpp", "ds1000", "humaneval_plus"]:
                reduce_prompt_suffix = "\n```python\n"
            else:
                reduce_prompt_suffix = "\nFinal Answer:"
        else:
            reduce_prompt_suffix = f"\nAnswer the question based on the above information. Answer concisely.\nAnswer:"

        reduce_prompt_suffix = build_suffix(self.model_name, reduce_prompt_suffix)
        reduce_prompt_suffix_ids = self.tokenizer(
            reduce_prompt_suffix, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        self.suffix_len = reduce_prompt_suffix_ids.shape[1]
        current_length = self.prefix_len
        information_ids = torch.empty(
            (1, 0), dtype=torch.int64, device=self.device
        )
        self.ctx_indices_list = []
        self.addition_indices_list = []
        for i, sub_ids in enumerate(information_ids_list):
            self.addition_indices_list.append([
                x for x in self.recompute_index_list[i]
            ])
            self.recompute_index_list[i] = [
                x + current_length for x in self.recompute_index_list[i]
            ]
            self.ctx_indices_list.append([current_length, current_length + sub_ids.shape[1] -1])
            current_length += sub_ids.shape[1]
            information_ids = torch.cat(tensors=[information_ids, sub_ids], dim=-1)

        partial_reduce_prompt_ids = torch.cat([self.prefix_ids, information_ids], dim=-1)
        self.partial_len = partial_reduce_prompt_ids.shape[1]

        self.full_input_ids = torch.cat(
            [partial_reduce_prompt_ids, reduce_prompt_suffix_ids], dim=-1
        ).contiguous()

        full_past_key_values = DynamicCache()
        for l in range(self.full_num_layers):
            key_caches = [self.prefix_cache.key_cache[l].clone()]
            value_caches = [self.prefix_cache.value_cache[l].clone()]

            for info_cache in self.information_cache_list:
                key_caches.append(info_cache.key_cache[l].clone())
                value_caches.append(info_cache.value_cache[l].clone())

            full_past_key_values.key_cache.append(torch.cat(key_caches, dim=-2))
            full_past_key_values.value_cache.append(torch.cat(value_caches, dim=-2))

        position_ids = torch.tensor(list(range(self.partial_len)), device=self.device).unsqueeze(0)
        self.final_cache_with_rope = kv_cache_apply_rope(
            full_past_key_values, position_ids, self.rope_module
        )
        self.recompute_indices = list(range(self.prefix_len, self.partial_len))

        self.logger.info(f"Precomputation complete. Total input length: {self.full_input_ids.shape[1]}")
        return True

    def _run_single_generation(self, mode: str, current_params: Dict[str, Any]) -> dict:
        """
        Execute a single model.generate call in memory.
        (Uses imported 'kv_cache_apply_rope' and 'RelayCache')
        """
        st_layer = current_params["det_layer"]

        self.logger.info(f"--> Executing: {mode} | det_layer={st_layer} | Params: {current_params}")
        profile_lower = mode == "low"
        profile_upper = mode == "high"

        layer_idx = st_layer
        reuse_layer_idx = layer_idx if profile_lower else current_params["reuse_hidden_layer"]
        end_layer_val = self.full_num_layers
        # Layer sweep profiling: no deviation filtering at detection layer.
        tau_dev_val = 0.0

        current_layer_hidden_states = self.all_layer_hidden_states[reuse_layer_idx]
        relay_cache = RelayCache(
            self.final_cache_with_rope,
            recompute_indices = self.recompute_indices,
            blank_absolute_indices = [],
            blank_relative_indices = [],
            relay_absolute_indices = self.recompute_indices,
            relay_relative_indices = list(range(len(self.recompute_indices))),
            
            recompute_label = True,
            det_layer = layer_idx, 
            end_layer = end_layer_val,
            
            tau_dev = tau_dev_val,
            tau_inf = current_params["tau_inf"],
            prefix_tokens = current_params["prefix_tokens"],
            suffix_tokens = current_params["suffix_tokens"],
            
            reuse_hidden_layer = reuse_layer_idx,
            store_hidden_layer = None,
            execution_mode = "relaycaching",
            additional_indices = self.addition_indices_list,
            relay_chunk_indices = self.ctx_indices_list,
            profile_lower_layer = profile_lower,
            profile_upper_layer = profile_upper,
            reusable_hidden_states_list = current_layer_hidden_states,
        )
        with torch.no_grad():
            outputs = self.model.generate(
                self.full_input_ids,
                max_new_tokens=self.config.get("max_new_tokens", 2),
                use_cache=True,
                do_sample = False,
                past_key_values=relay_cache,
                return_dict_in_generate=True,
                output_attentions=False, # temporarily disabled for performance
            )
        
        return outputs
    
    def _run_experiment_reuse(self, mode: str, fixed_start_layer: Optional[int] = None) -> Dict[Any, Any]:
        self.logger.info(f"--- Starting {mode.upper()} Bound Layer Sweep ---")
        all_generated_data = {}

        for val in self.sweep_values:
            current_params = self.default_params.copy()
            layer_val = int(val)

            if mode == "low":
                current_params["det_layer"] = layer_val
                current_params["end_layer"] = self.num_layers
            elif mode == "high":
                current_params["det_layer"] = layer_val
                current_params["end_layer"] = self.num_layers
                if fixed_start_layer is not None:
                    if layer_val < fixed_start_layer:
                        self.logger.info(
                            f"Skipping invalid layer {layer_val} < fixed_start_layer {fixed_start_layer}"
                        )
                        continue
                    current_params["reuse_hidden_layer"] = fixed_start_layer

            range_str = f"[{current_params['det_layer']}, {current_params['end_layer']}]"
            self.logger.info(f"    Sweep Val: {val} | Range: {range_str}")

            outputs = self._run_single_generation(mode, current_params)

            generated_cache_data = extract_kv_with_tokens(
                outputs,
                self.partial_len,
                self.suffix_len
            )
            
            all_generated_data[val] = generated_cache_data
            
        # Efficient Scan mode: fill NaN placeholders for skipped layers to keep
        # sweep_values and all_generated_data keys in 1:1 correspondence for plotting
        if mode == "high" and fixed_start_layer is not None:
            for val in self.sweep_values:
                if val not in all_generated_data:
                    layer_val = int(val)
                    if layer_val < fixed_start_layer:
                        self.logger.info(f"Adding NaN placeholder for skipped layer {layer_val} (< fixed_start_layer {fixed_start_layer})")
                        all_generated_data[val] = None
                    else:
                        self.logger.warning(f"Layer {layer_val} unexpectedly missing from generated data")

        return all_generated_data


        
    def _analyze_results(self, mode: str, generated_data: Dict[int, Any], 
                         no_repair_sim: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
        """
        Analyze data stored in memory.

        Args:
            mode: mode ("low" or "high")
            generated_data: generated data
            no_repair_sim: no-repair baseline similarity (kept for interface compatibility, currently unused)

        Returns:
            result dictionary containing absolute values
        """
        self.logger.info(f"--- Analyzing {mode.upper()} Bound Results (In-Memory) ---")

        if self.reference_cache is None:
            self.logger.error("Gold baseline (self.reference_cache) missing. Cannot analyze results.")
            raise RuntimeError("Gold baseline cache missing.")
        ref_kv = self.reference_cache

        results_lists = {
            "key_cosine": [], "value_cosine": [],
            "key_norm": [], "value_norm": []
        }

        for val in self.sweep_values:
            recomputed_kv = generated_data.get(val)

            if recomputed_kv is None:
                self.logger.warning(f"Val {val} result missing. Recorded as NaN.")
                for key in results_lists:
                    results_lists[key].append(float('nan'))
                continue

            overall_metrics = calculate_overall_similarity(ref_kv, recomputed_kv)

            if not overall_metrics:
                self.logger.error("Similarity computation failed. Recorded as NaN.")
                for key in results_lists:
                    results_lists[key].append(float('nan'))
            else:
                key_cos = overall_metrics.get("key_cosine_similarity", float('nan'))
                value_cos = overall_metrics.get("value_cosine_similarity", float('nan'))
                key_norm = overall_metrics.get("key_norm_similarity", float('nan'))
                value_norm = overall_metrics.get("value_norm_similarity", float('nan'))

                results_lists["key_cosine"].append(key_cos)
                results_lists["value_cosine"].append(value_cos)
                results_lists["key_norm"].append(key_norm)
                results_lists["value_norm"].append(value_norm)

        return results_lists

    def _plot_curves(self, results: Dict[str, List[float]], mode: str):
        self.logger.info(f"--- Plotting {mode.upper()} Bound Curve ---")

        num_points = len(results.get("key_cosine", []))
        if num_points == 0:
            self.logger.warning(f"No data points for {mode} curve. Skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        if num_points <= len(self.sweep_values):
            x_axis = self.sweep_values[:num_points]
        else:
            x_axis = list(range(num_points))

        if not x_axis:
            self.logger.warning(f"x_axis is empty for {mode} curve. Skipping plot.")
            plt.close(fig)
            return

        ax.plot(x_axis, results["key_cosine"],
                linewidth=2.5, color='#1f77b4', marker='s',
                markersize=5, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor='#1f77b4',
                label='Key Cosine Similarity')

        ax.plot(x_axis, results["value_cosine"],
                linewidth=2.5, color='#2ca02c', marker='^',
                markersize=5, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor='#2ca02c',
                label='Value Cosine Similarity')

        ax.plot(x_axis, results["key_norm"],
                linewidth=2.5, color='#ff7f0e', marker='D',
                markersize=5, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor='#ff7f0e',
                label='Key Norm Similarity', linestyle='--')

        ax.plot(x_axis, results["value_norm"],
                linewidth=2.5, color='#d62728', marker='v',
                markersize=5, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor='#d62728',
                label='Value Norm Similarity', linestyle='--')

        scan_mode_info = "Full Scan"
        title = (
            f"Similarity vs. Layer Index ({mode}) [{scan_mode_info}]\n"
            f"([{self.default_params['det_layer']}, {self.default_params['end_layer']}], Data Len: {self.data_len})"
        )
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xlabel("Layer Index", fontweight='bold', fontsize=12)
        ax.set_ylabel("Overall Similarity", fontweight='bold', fontsize=12)

        ax.set_xticks(x_axis)
        ax.set_xlim(-0.5, self.num_layers - 0.5)

        if len(x_axis) > 20:
            ax.tick_params(axis='x', rotation=90)
        else:
            ax.tick_params(axis='x', rotation=0)

        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.set_axisbelow(True)
        ax.grid(True, which="both", zorder=0, color='#E0E0E0', linestyle='-', linewidth=0.8)
        plt.tight_layout()

        png_save_path = self.plot_dir / f"similarity_curve_{mode}.png"
        fig.savefig(png_save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"PNG Plot saved to: {png_save_path}")

        pdf_save_path = self.plot_dir / f"similarity_curve_{mode}.pdf"
        fig.savefig(pdf_save_path, format='pdf', bbox_inches='tight')
        self.logger.info(f"PDF Plot saved to: {pdf_save_path}")
        
        plt.close(fig)

    def _plot_profile_selection(
        self,
        layer_similarities: List[Dict],
        L_start: int,
        L_end: int,
        L_det: int,
    ):
        """Plot U-shape profile and Spearman/alpha curves with selected layer markers."""
        if plot_layer_profile_selection is None:
            self.logger.warning("plot_layer_profile_selection unavailable; skip profile plot.")
            return
        if not layer_similarities:
            return
        model_label = getattr(self, "llm_name", None) or str(getattr(self, "model_path", "Model"))
        pdf_path = self.plot_dir / "layer_profile_selection.pdf"
        self.logger.info(f"--- Plotting layer profile selection → {pdf_path}")
        plot_layer_profile_selection(
            layer_similarities=layer_similarities,
            L_start=L_start,
            L_end=L_end,
            L_det=L_det,
            correlations=self._spearman_correlations_cache,
            second_order_diffs=self._second_order_diffs_cache,
            metric=self.layer_selection_metric,
            lower_threshold=self.layer_lower_threshold,
            model_name=model_label,
            save_path=str(pdf_path),
        )

    def _plot_combined_upper_lower_curves(self, low_results: Dict[str, List[float]], high_results: Dict[str, List[float]]):
        """
        Plot Value Cosine curves including both Upper Bound and Lower Bound scans,
        similar to the layer deviation plot in observation experiments, annotating shallow, middle, and deep layers.

        Args:
            low_results: Lower Bound sweep results
            high_results: Upper Bound sweep results
        """
        self.logger.info("--- Plotting Combined Upper & Lower Bound Curves ---")

        num_low_points = len(low_results.get("value_cosine", []))
        num_high_points = len(high_results.get("value_cosine", []))

        if num_low_points == 0 and num_high_points == 0:
            self.logger.warning("No data points for combined curves. Skipping plot.")
            return

        x_low = self.sweep_values[:num_low_points]
        x_high = self.sweep_values[:num_high_points]

        fig, ax = plt.subplots(figsize=(14, 7))

        shallow_end = max(1, self.num_layers // 8)
        deep_start = min(self.num_layers - 1, self.num_layers * 3 // 4)

        ax.axvspan(-0.5, shallow_end - 0.5, color='#E8F5E9', label='Shallow Layers')
        ax.axvspan(shallow_end - 0.5, deep_start - 0.5, color='#FFEBEE', label='Middle Layers')
        ax.axvspan(deep_start - 0.5, self.num_layers - 0.5, color='#E3F2FD', label='Deep Layers')
        ax.plot(x_low, low_results["value_cosine"],
                linewidth=2.5, color='#2ca02c', marker='^',
                markersize=5, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor='#2ca02c',
                label='Lower Bound (Rec[0, L_end])')

        ax.plot(x_high, high_results["value_cosine"],
                linewidth=2.5, color='#d62728', marker='s',
                markersize=5, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor='#d62728',
                label='Upper Bound (Rec[L_start, Max])')

        scan_mode_info = "Full Scan"
        title = (
            f"Value Cosine Similarity: Upper & Lower Bound Sweep [{scan_mode_info}]\\n"
            f"([{self.default_params['det_layer']}, {self.default_params['end_layer']}], "
            f"Data Len: {self.data_len}, Model: {self.model_name.split('/')[-1]})"
        )
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xlabel("Layer Index", fontweight='bold', fontsize=12)
        ax.set_ylabel("Value Cosine Similarity", fontweight='bold', fontsize=12)

        ax.set_xlim(-0.5, self.num_layers - 0.5)
        ax.set_xticks(x_low)

        if len(x_low) > 20:
            ax.tick_params(axis='x', rotation=90)

        all_sims = low_results["value_cosine"] + high_results["value_cosine"]
        if all_sims:
            valid_sims = [s for s in all_sims if not np.isnan(s)]
            if valid_sims:
                min_sim = min(valid_sims)
                max_sim = max(valid_sims)
                y_range = max_sim - min_sim
                y_min = max(0.0, min_sim - y_range * 0.1)
                y_max = min(1.0, max_sim + y_range * 0.05)
                ax.set_ylim([y_min, y_max])
            else:
                self.logger.warning("All similarity values are NaN. Using default y-axis limits.")
                ax.set_ylim([0.95, 1.0])
        else:
            self.logger.warning("No combined similarity data for y-axis limits. Using default.")

        ax.set_axisbelow(True)
        ax.grid(True, which="both", zorder=0, color='#E0E0E0', linestyle='-', linewidth=0.8)
        ax.legend(loc='best', fontsize=9, ncol=2)
        plt.tight_layout()

        png_save_path = self.plot_dir / f"combined_upper_lower_curves.png"
        fig.savefig(png_save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"PNG Plot saved to: {png_save_path}")

        pdf_save_path = self.plot_dir / f"combined_upper_lower_curves.pdf"
        fig.savefig(pdf_save_path, format='pdf', bbox_inches='tight')
        self.logger.info(f"PDF Plot saved to: {pdf_save_path}")
        
        plt.close(fig)



    def _save_experiment_results(
        self,
        low_results: Dict[str, List[float]],
        high_results: Dict[str, List[float]],
        final_L_start: int = 0,
        final_L_end: int = 0,
        final_L_det: int = 0,
        merged_layer_similarities: Optional[List[Dict]] = None,
        num_calibration_samples: int = 1,
        verbose: bool = True,
    ):
        """
        Save experiment results to JSON (aligned with observation experiment_report format).
        """
        self.logger.info("--- Saving experiment results to JSON ---")

        experiment_report = {
            "experiment_config": {
                "model_name": self.model_name,
                "device": self.device_str,
                "dataset_path": self.config.get("eval_dataset_path", ""),
                "st_idx": self.st_idx,
                "data_len": self.data_len,
                "ctx_num": self.ctx_num,
                "num_layers": self.num_layers,
                "run_layer_sweep": self.run_layer_sweep,
                "layer_selection_metric": self.layer_selection_metric,
                "layer_selection_hyperparameters": {
                    "layer_lower_threshold": self.layer_lower_threshold,
                    "layer_upper_sigma_multiplier": self.layer_upper_sigma_multiplier,
                    "layer_upper_consecutive_count": self.layer_upper_consecutive_count,
                    "layer_upper_min_samples": self.layer_upper_min_samples,
                    "right_tail_layers": self.right_tail_layers,
                    "layer_override": self.layer_override,
                },
                "default_params": self.default_params,
            },
            "auto_selected_layer_range": {
                "lower_bound": int(final_L_start),
                "upper_bound": int(final_L_end),
                "det_layer": int(final_L_det),
                "method": "decoding_vs_prefill_u_shape_avg_curve",
                "note": (
                    "layer_similarities averaged over calibration samples; "
                    "lower_bound/upper_bound are inclusive L_start/L_end (Relay --end_layer)"
                ),
            },
            "num_calibration_samples": num_calibration_samples,
            "layer_similarities": merged_layer_similarities or [],
            "timestamp": datetime.now().isoformat(),
        }

        if self.run_layer_sweep and low_results:
            experiment_report["diagnostic_sweep_vs_gold"] = {
                "lower_bound_sweep": low_results,
                "upper_bound_sweep": high_results,
                "curve_L_min": int(self._find_similarity_range(low_results, 0.99, mode='L_min')),
                "curve_L_max": int(self._find_similarity_range(high_results, 0.99, mode='L_max')),
            }

        results_path = self.base_output_dir / "experiment_report.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Experiment report saved to: {results_path}")

    def _find_layer_range(self, layer_similarities: List[float],
                          lower_threshold: float = 0.99,
                          verbose: bool = True) -> Tuple[int, int]:
        """
        Determine layer range [L_start, L_end] using the algorithm from Paper Section 4.2.

        Delegates to layer_selection.auto_select_middle_layer_range(), which implements:
        - L_start: last layer with similarity >= threshold
        - L_end: U-shape right-side stable region detection (Welford online statistics)

        Args:
            layer_similarities: per-layer value cosine similarity list
            lower_threshold: L_start threshold (default 0.99)
            verbose: whether to print detailed info

        Returns:
            (L_start, L_end): layer range
        """
        if auto_select_middle_layer_range is not None:
            return auto_select_middle_layer_range(
                layer_similarities=layer_similarities,
                lower_threshold=lower_threshold,
                metric="value_cosine" if isinstance(layer_similarities[0], dict) else None,
                verbose=verbose
            )
        else:
            # Fallback: simple threshold method
            self.logger.warning("layer_selection module not available, using simple threshold fallback.")
            return self._find_similarity_range_fallback(layer_similarities, lower_threshold)
    
    def _find_similarity_range_fallback(self, similarities: List[float], threshold: float = 0.99) -> Tuple[int, int]:
        """
        Fallback: simple threshold-based layer range detection.
        Used only when layer_selection module is not available.
        """
        num_layers = len(similarities)
        
        # L_start: last layer with sim >= threshold
        L_start = 0
        for i, sim in enumerate(similarities):
            if not np.isnan(sim) and sim >= threshold:
                L_start = i
        
        # L_end: first layer (after valley) with sim >= threshold
        L_end = num_layers
        valley = int(np.nanargmin(similarities))
        for i in range(valley + 1, num_layers):
            if not np.isnan(similarities[i]) and similarities[i] >= threshold:
                L_end = i
                break
        
        return L_start, L_end

    def _find_similarity_range(self, results: Dict[str, List[float]], target_sim: float = 0.99, mode: str = 'L_min') -> int:
        """
        Legacy interface: find layer index satisfying similarity threshold.
        
        Delegates to _find_layer_range() for the full algorithm when possible,
        but maintains backward compatibility for code that calls with mode='L_min'/'L_max'.
        
        Args:
            mode ('L_min'/'L_max'): 
                - 'L_min': Find last layer with sim >= threshold (recomputation start)
                - 'L_max': Find first layer (after valley) with sim >= threshold (recomputation end)
        """
        sim_scores = results['value_cosine']
        num_scores = len(sim_scores)
        
        # Filter NaN values
        valid_scores = [(i, score) for i, score in enumerate(sim_scores) if not np.isnan(score)]

        if mode == 'L_min':
            for i, score in valid_scores:
                if score < target_sim:
                    if i > 0:
                        return int(self.sweep_values[i - 1])
                    else:
                        return 0
            return int(self.sweep_values[0]) if len(self.sweep_values) > 0 else 0

        elif mode == 'L_max':
            for i, score in valid_scores:
                if score >= target_sim:
                    if i > 0:
                        return int(self.sweep_values[i - 1])
                    else:
                        return 0
            return int(self.sweep_values[-1]) if len(self.sweep_values) > 0 else self.num_layers - 1
        
        raise ValueError("Invalid mode specified for _find_similarity_range.")
    
    def run(self):
        """
        Run the complete integrated Profiling workflow (supports multiple data samples).
        """
        self.logger.info("Starting integrated Layer Sensitivity Profiling workflow...")

        try:
            self._setup_directories()

            map_path = Path(self.config["map_results_path"])
            needs_map_run = not map_path.exists()

            self._load_model_and_tokenizer()
            self._load_profiling_data()

            if self.run_layer_sweep:
                if not self.sweep_values:
                    self.sweep_values = list(range(self.num_layers))
                    self.logger.info(
                        f"Diagnostic sweep layer indices: {self.sweep_values}"
                    )

            if needs_map_run or self.config.get("must_run_map_phase", True):
                self.logger.info("Phase 1: Running Map phase (generating summaries)")
                self._run_map_phase()
            else:
                self.logger.info("Phase 1: Loading Map results from file")
                self._load_map_results_from_file()

            all_high_bound_results = []
            all_low_bound_results = []
            per_sample_layer_sims: List[List[Dict]] = []
            per_sample_token_devs: List[Dict[int, np.ndarray]] = []
            valid_count = 0

            end_idx = self.st_idx + self.data_len

            for current_data_idx in range(self.st_idx, end_idx):
                self.logger.info(f"\n{'='*40}")
                self.logger.info(f">>> Processing Data Index: {current_data_idx} <<<")
                self.logger.info(f"{'='*40}")

                try:
                    torch.cuda.empty_cache()

                    is_valid_data = self._prepare_all_caches(data_idx=current_data_idx)

                    if not is_valid_data:
                        self.logger.warning(f"Skipping Data {current_data_idx} due to empty context/filter.")
                        continue

                    layer_sims = self._compute_layer_similarities_observation(current_data_idx)
                    per_sample_layer_sims.append(layer_sims)

                    self._collect_token_deviations_dec_prefill(ctx_idx=0)
                    if self._token_deviations_cache:
                        per_sample_token_devs.append(dict(self._token_deviations_cache))

                    if self.run_layer_sweep:
                        self.logger.info(
                            f"[Data {current_data_idx}] Optional relay sweep (diagnostic, vs gold)..."
                        )
                        self.reference_cache = self._run_gold_baseline()
                        _, no_repair_sim_low = self._run_no_repair_baseline(mode="low")
                        low_data = self._run_experiment_reuse(mode="low")
                        results_low = self._analyze_results(
                            mode="low", generated_data=low_data, no_repair_sim=no_repair_sim_low,
                        )
                        all_low_bound_results.append(results_low)

                        _, no_repair_sim_high = self._run_no_repair_baseline(mode="high")
                        high_data = self._run_experiment_reuse(mode="high", fixed_start_layer=0)
                        results_high = self._analyze_results(
                            mode="high", generated_data=high_data, no_repair_sim=no_repair_sim_high,
                        )
                        all_high_bound_results.append(results_high)

                    valid_count += 1

                except Exception as e:
                    self.logger.error(f"Error processing Data {current_data_idx}: {e}", exc_info=True)
                    continue

            if valid_count > 0:
                self.logger.info("="*80)
                self.logger.info(f"Calibration complete (valid: {valid_count}/{self.data_len}).")

                merged_layer_sims = self._merge_layer_similarity_lists(per_sample_layer_sims)
                self.logger.info(
                    f"Profiling on averaged layer curve "
                    f"({len(per_sample_layer_sims)} calibration sample(s))."
                )

                if self.layer_override:
                    final_L_start = int(self.layer_override["start"])
                    final_L_end = int(self.layer_override["end"])
                    final_L_det = int(self.layer_override["det"])
                    self._spearman_correlations_cache = {}
                    self._second_order_diffs_cache = {}
                    self.logger.info(
                        "Using CLI layer override (--start-layer / --det-layer / --end-layer); "
                        "skipping auto-selection."
                    )
                else:
                    final_L_start = 0
                    final_L_end = self.num_layers - 1
                    if auto_select_middle_layer_range is not None and merged_layer_sims:
                        final_L_start, final_L_end = auto_select_middle_layer_range(
                            layer_similarities=merged_layer_sims,
                            lower_threshold=self.layer_lower_threshold,
                            upper_sigma_multiplier=self.layer_upper_sigma_multiplier,
                            upper_consecutive_count=self.layer_upper_consecutive_count,
                            upper_min_samples=self.layer_upper_min_samples,
                            right_tail_layers=self.right_tail_layers,
                            metric=self.layer_selection_metric,
                            verbose=True,
                        )
                        self._spearman_correlations_cache = {}
                    self._second_order_diffs_cache = {}
                    final_L_det = final_L_start + 1
                    if auto_select_detection_layer is not None and per_sample_token_devs:
                        det_out = auto_select_detection_layer(
                            per_sample_token_deviations=per_sample_token_devs,
                            L_start=final_L_start,
                            L_end=final_L_end,
                            verbose=True,
                            return_diagnostics=True,
                        )
                        final_L_det, self._spearman_correlations_cache, self._second_order_diffs_cache = det_out
                    elif per_sample_token_devs:
                        self.logger.info(f"  [L_det Fallback] Using L_start + 1 = {final_L_det}")
                    else:
                        self.logger.info(f"  [L_det Fallback] No token deviations; L_start + 1 = {final_L_det}")

                self.logger.info("--- Layer range recommendation (Decoding vs Prefill) ---")
                self.logger.info(
                    f"  L_start (stlayer) = {final_L_start} (deepest sim>=τ); "
                    f"critical range [{final_L_start}, {final_L_end}] inclusive "
                    f"(--start_layer {final_L_start} --end_layer {final_L_end})."
                )

                if self.run_layer_sweep and all_low_bound_results:
                    target_similarity = 0.99
                    avg_low = self._aggregate_results(all_low_bound_results)
                    avg_high = self._aggregate_results(all_high_bound_results)
                    sweep_L_min = self._find_similarity_range(avg_low, target_similarity, mode='L_min')
                    sweep_L_max = self._find_similarity_range(avg_high, target_similarity, mode='L_max')
                    self.logger.info(
                        f"  [Diagnostic sweep vs gold] Range: [{sweep_L_min}, {sweep_L_max}] "
                        "(not used for final recommendation)"
                    )
                    self._plot_curves(avg_high, mode="avg_high")
                    self._plot_curves(avg_low, mode="avg_low")
                    self._plot_combined_upper_lower_curves(avg_low, avg_high)
                else:
                    avg_low, avg_high = {}, {}

                self.logger.info("=" * 60)
                self.logger.info("  FINAL LAYER RANGE RECOMMENDATION (Paper Section 4.2)")
                self.logger.info("=" * 60)
                self.logger.info(
                    f"  L_start (stlayer) = {final_L_start}  "
                    f"(deepest layer with sim >= τ; shallow [0, {max(0, final_L_start - 1)}])"
                )
                self.logger.info(
                    f"  --start_layer = {final_L_start}  "
                    f"(Paper L_start; hidden relay + start of critical range)"
                )
                self.logger.info(f"  L_end   = {final_L_end}  (--end_layer {final_L_end})")
                self.logger.info(f"  L_det   = {final_L_det}  (--det_layer {final_L_det})")
                self.logger.info("=" * 60)

                self._plot_profile_selection(
                    layer_similarities=merged_layer_sims,
                    L_start=final_L_start,
                    L_end=final_L_end,
                    L_det=final_L_det,
                )

                print("\n" + "=" * 60)
                print("  RECOMMENDED PARAMETERS (observation-style):")
                print(f"    --start_layer {final_L_start}  (L_start / stlayer)")
                print(f"    --det_layer {final_L_det}")
                print(f"    --end_layer {final_L_end}")
                print("=" * 60 + "\n")

                self._save_experiment_results(
                    avg_low, avg_high,
                    final_L_start=final_L_start,
                    final_L_end=final_L_end,
                    final_L_det=final_L_det,
                    merged_layer_similarities=merged_layer_sims,
                    num_calibration_samples=len(per_sample_layer_sims),
                )
            else:
                self.logger.warning("No data samples were successfully processed.")

        except Exception as e:
            self.logger.critical(f"Fatal error in profiling workflow: {e}", exc_info=True)


def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_layer_auto_selection_args(parser: argparse.ArgumentParser) -> None:
    """
    CLI for algorithms that auto-pick L_start, L_end (U-shape) and L_det (Spearman/α).
    All of these are consumed by LayerRangeProfiler via profiler_config.
    """
    group = parser.add_argument_group(
        "Layer auto-selection method",
        description=(
            "Hyperparameters for auto_select_middle_layer_range (L_start, L_end) and "
            "auto_select_detection_layer (L_det). "
            "To skip auto-detection, set all three overrides below."
        ),
    )
    group.add_argument(
        "--tau-low",
        dest="layer_lower_threshold",
        type=float,
        default=0.99,
        metavar="τ_low",
        help="Lower similarity threshold for L_start (default 0.99).",
    )
    group.add_argument(
        "--right-tail-layers",
        dest="right_tail_layers",
        type=int,
        default=5,
        metavar="T",
        help="Number of deepest layers used to estimate stable-region stats (default 5).",
    )
    group.add_argument(
        "--stable-multiplier",
        dest="layer_upper_sigma_multiplier",
        type=float,
        default=0.5,
        metavar="λ",
        help="Stable multiplier λ for L_end entry threshold: mean - λ·std (default 0.5).",
    )
    group.add_argument(
        "--min-rise-samples",
        dest="layer_upper_min_samples",
        type=int,
        default=3,
        metavar="N_min",
        help="Minimum rise samples (Welford) before L_end detection (default 3).",
    )
    group.add_argument(
        "--consecutive-stables",
        dest="layer_upper_consecutive_count",
        type=int,
        default=2,
        metavar="C",
        help="Consecutive stable layers required for L_end (default 2).",
    )
    group.add_argument(
        "--layer-selection-metric",
        type=str,
        default="value_cosine",
        choices=["key_cosine", "value_cosine", "key_norm", "value_norm"],
        help="Per-layer curve metric for U-shape (L_start / L_end).",
    )
    group.add_argument(
        "--start-layer",
        type=int,
        default=None,
        help="Manual L_start; requires --det-layer and --end-layer (disables auto-select).",
    )
    group.add_argument(
        "--det-layer",
        type=int,
        default=None,
        help="Manual L_det; requires --start-layer and --end-layer.",
    )
    group.add_argument(
        "--end-layer",
        type=int,
        default=None,
        help="Manual inclusive L_end; requires --start-layer and --det-layer.",
    )


def _layer_selection_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "layer_lower_threshold": args.layer_lower_threshold,
        "layer_upper_sigma_multiplier": args.layer_upper_sigma_multiplier,
        "layer_upper_consecutive_count": args.layer_upper_consecutive_count,
        "layer_upper_min_samples": args.layer_upper_min_samples,
        "right_tail_layers": args.right_tail_layers,
        "layer_selection_metric": args.layer_selection_metric,
    }
    if (
        args.start_layer is not None
        or args.det_layer is not None
        or args.end_layer is not None
    ):
        if (
            args.start_layer is None
            or args.det_layer is None
            or args.end_layer is None
        ):
            raise ValueError(
                "Manual layer override requires all of "
                "--start-layer, --det-layer, --end-layer"
            )
        cfg["layer_override"] = {
            "start": args.start_layer,
            "det": args.det_layer,
            "end": args.end_layer,
        }
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Layer-Range Profiler: "
                    "Determine critical layer range [L_start, L_end] and detection layer L_det.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # --- Core arguments ---
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to calibration dataset")
    parser.add_argument("--dataset-type", type=str, default="wikimqa",
                        choices=["wikimqa", "gsm8k", "math", "humaneval", "mmlu"],
                        help="Dataset type (determines prompt template)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on (e.g., cuda:0)")

    # --- Data selection ---
    parser.add_argument("--st-idx", type=int, default=0,
                        help="Starting sample index")
    parser.add_argument("--data-len", type=int, default=3,
                        help="Number of calibration samples (more = more robust, slower)")
    parser.add_argument("--ctx-num", type=int, default=1,
                        help="Number of contexts per sample (WikiMQA only)")

    _add_layer_auto_selection_args(parser)

    # --- Profiler parameters ---
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="./profiling_results",
                        help="Output directory for profiling results and plots")

    # --- Advanced: override auto-detection ---
    parser.add_argument("--domain", type=str, default=None,
                        help="Prompt domain override (auto-detected from dataset-type if not set)")
    parser.add_argument("--role", type=str, default=None,
                        help="Agent role override (uses domain default if not set)")
    parser.add_argument("--num-layers", type=int, default=-1,
                        help="Number of model layers (-1 = auto-detect from model config)")
    parser.add_argument("--run-layer-sweep", action="store_true",
                        help="Optional diagnostic: relay sweep vs gold (legacy layer_profiler)")

    args = parser.parse_args()
    try:
        layer_sel_cfg = _layer_selection_config_from_args(args)
    except ValueError as e:
        parser.error(str(e))

    # Set seed
    seed_everything(args.seed)

    # Auto-detect domain from dataset-type if not specified
    domain = args.domain
    role = args.role
    if domain is None:
        domain_map = {
            "wikimqa": None,
            "gsm8k": "gsm8k",
            "math": "math",
            "humaneval": "humaneval",
            "mmlu": None,
        }
        domain = domain_map.get(args.dataset_type)

    # Sanitize model name for file paths
    sanitized_model = args.model.replace("/", "_")
    map_results_path = f"{args.output}/mapres_{sanitized_model}_st{args.st_idx}_len{args.data_len}_ctx{args.ctx_num}.json"

    # Build profiler config
    profiler_config = {
        # Data/paths
        "st_idx": args.st_idx,
        "data_len": args.data_len,
        "ctx_num": args.ctx_num,
        "eval_dataset_path": args.data,
        "map_results_path": map_results_path,
        "base_output_dir": args.output,

        # Model/environment
        "model_name": args.model,
        "device": args.device,
        "num_layers": args.num_layers,  # -1 = auto-detect from model config

        # Dataset support
        "dataset_type": args.dataset_type,
        "domain": domain,
        "role": role,

        # Script behavior
        "max_new_tokens": 2,
        "is_prefix": True,
        "not_use_filter": False,
        "must_run_map_phase": True,
        "run_layer_sweep": args.run_layer_sweep,

        # Layer selection hyperparameters from CLI
        **layer_sel_cfg,

        # Default parameters
        "tau_dev": 0.0,
        "tau_inf": 0.0,
        "prefix_tokens": 0,
        "suffix_tokens": 0,
        "det_layer": 0,
        "end_layer": 32,
        "reuse_hidden_layer": 0,
    }

    print("=" * 60)
    print("  Layer-Range Profiler")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Dataset:     {args.data} ({args.dataset_type})")
    print(f"  Samples:     {args.data_len} (starting at idx {args.st_idx})")
    print(f"  Device:      {args.device}")
    print(f"  Output:      {args.output}")
    print("  Layer selection (U-shape + L_det):")
    print(f"    metric={args.layer_selection_metric}")
    print(f"    τ_low={args.layer_lower_threshold}  T={args.right_tail_layers}  "
          f"λ={args.layer_upper_sigma_multiplier}  "
          f"N_min={args.layer_upper_min_samples}  C={args.layer_upper_consecutive_count}")
    if args.start_layer is not None:
        print(
            f"    override: start={args.start_layer} det={args.det_layer} "
            f"end={args.end_layer}"
        )
    print("=" * 60)
    print()

    profiler = LayerRangeProfiler(config=profiler_config)
    profiler.run()


if __name__ == "__main__":
    main()
