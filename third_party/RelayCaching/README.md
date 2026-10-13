[![arXiv](https://img.shields.io/badge/arXiv-2603.13289-b31b1b.svg)](https://arxiv.org/abs/2603.13289)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
<h1>RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse</h1>
</div>


<p align="center">
    <br>
    <img src="assets/main_arch.png" width="100%"/>
    <br>
</p>

## Introduction

This is the official code repository for the paper **"RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse"** ([arXiv:2603.13289](https://arxiv.org/abs/2603.13289)).

In multi-agent LLM systems, upstream agents' outputs are embedded into downstream prompts, causing **cascading redundant prefill computation** that significantly increases KV-cache memory usage and Time-To-First-Token (TTFT). Existing methods based on prefix caching or pre-computed caching fail in this setting due to non-prefix positions and dynamic content.

**RelayCaching** is a training-free inference method that directly reuses decoding-phase KV caches from upstream agents in downstream prefill phases. Our key insight is that decoding KV caches maintain high alignment with full-prefill counterparts despite prefix variation. Residual deviations follow systematic patterns: a U-shaped layer-wise similarity profile (concentrating in middle layers) and sparse, inter-layer-correlated token-wise deviations. By selectively rectifying KV caches only at these critical layers and token positions, RelayCaching preserves generation quality with minimal overhead.

### Key Features

- **Layer-Range Profiler**: Exploits the U-shaped layer-wise similarity profile to identify a critical layer range $[L_{\text{start}}, L_{\text{end}}]$ where deviations concentrate, and locates a detection layer $L_{\text{det}}$ via inter-layer correlation analysis.
- **Deviation-Based Token Selection**: At $L_{\text{det}}$, identifies tokens whose value vectors deviate most from full-prefill counterparts using adaptive mean-relative thresholding ($\tau_{\text{dev}} \cdot \mu_{\text{dev}}$).
- **Influence-Based Token Selection**: Prioritizes tokens that receive high accumulated attention during decoding ($s_{\text{inf}}$), plus trailing suffix tokens ($K_{\text{suf}}$) to correct positional bias.
- **Hidden State Relay**: Stores intermediate hidden states at $L_{\text{start}}$ to bypass shallow-layer recomputation entirely.


## Installation

```bash
git clone https://github.com/YingshengGeng/RelayCaching.git
cd RelayCaching
uv venv relaycaching --python python3.12
source relaycaching/bin/activate
uv pip install -r requirements.txt 
```

## Project Structure

| Path | Role |
|------|------|
| `RelayCaching/` | Core package: inference (`llm/`), RelayCache (`utils/relay_utils.py`), agents, prompts, graph |
| `experiments/main/` | Benchmark evaluation (`run_mmlu.py`, `run_gsm8k.py`, `run_humaneval.py`) |
| `experiments/observation/` | KV alignment study (`run_observation.py`) |
| `experiments/profiler/` | Per-model layer calibration (`run_layer_profiler.py`) |
| `datasets_self/` | Bundled datasets |
| `assets/` | Architecture diagram |

> **Note**: The dataset directory is named `datasets_self` (not `datasets`) to avoid import conflicts with the HuggingFace `datasets` package.

## Quick Start

### Key Arguments

| Argument | Paper Symbol | Description |
|----------|:------------:|-------------|
| `--execution_mode` | — | Method to run: `relaycaching`, `cacheblend`, `epic`, or `kvcomm` |
| `--start_layer` | $L_{\text{start}}$ | Hidden state store/load layer. Layers `[L_start, L_det)` perform direct KV write (no token selection). Hidden states are stored here during Map and loaded here during Reuse |
| `--det_layer` | $L_{\text{det}}$ | Detection layer: token selection via deviation detection occurs here. Determines which tokens need selective recomputation |
| `--end_layer` | $L_{\text{end}}$ | End of the critical layer range. Selective recomputation stops after this layer; deeper layers use the corrected KV cache directly |
| `--tau_dev` | $\tau_{\text{dev}}$ | Deviation score threshold factor. Controls how many tokens are flagged for recomputation based on value deviation |
| `--tau_inf` | $\tau_{\text{inf}}$ | Influence score threshold factor. Controls additional token selection based on attention influence from already-selected tokens |
| `--suffix_tokens` | $L_{\text{suf}}$ | Number of trailing suffix tokens always recomputed for boundary correction |
| `--agent_nums` | — | Number of agents (excluding the decision agent) |
| `--llm_name` | — | HuggingFace model identifier |

## Run Experiments
Below are the commands to run main experiments from the paper.

<details>
<summary><b>RelayCaching</b></summary>

#### MMLU (LLaMA-3.1-8B)
```bash
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_relay_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode relaycaching \
    --start_layer 1 --det_layer 3 --end_layer 18 \
    --tau_dev 1.5 --tau_inf 1.45 --suffix_tokens 10
```

#### GSM8K (LLaMA-3.1-8B)
```bash
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_relay_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode relaycaching \
    --start_layer 1 --det_layer 3 --end_layer 18 \
    --tau_dev 1.5 --tau_inf 1.45 --suffix_tokens 10
```

#### HumanEval (Qwen2.5-Coder-7B)
```bash
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_qwen_relay_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode relaycaching \
    --start_layer 3 --det_layer 4 --end_layer 22 \
    --tau_dev 1.5 --tau_inf 1.45 --suffix_tokens 10
```

> **Sampling experiments** use stochastic decoding with 3 random seeds to measure mean and standard deviation. Each dataset has its own seed set.

```bash
# --- Sampling (seed=42) ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_relay_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode relaycaching \
    --start_layer 1 --det_layer 3 --end_layer 18 \
    --tau_dev 1.5 --tau_inf 1.45 --suffix_tokens 10 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_relay_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode relaycaching \
    --start_layer 1 --det_layer 3 --end_layer 18 \
    --tau_dev 1.5 --tau_inf 1.45 --suffix_tokens 10 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_qwen_relay_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode relaycaching \
    --start_layer 3 --det_layer 4 --end_layer 22 \
    --tau_dev 1.5 --tau_inf 1.45 --suffix_tokens 10 \
    --do-sample --seed 42 --temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.1
```

</details>

<details>
<summary><b>Baselines</b></summary>

#### Full Prefill (No KV Cache Reuse)
Full Prefill recomputes all KV caches from scratch (no reuse). We use EPIC with `epic_prefix_token_num=1000000` (exceeds any actual sequence length), so all tokens are treated as "prefix" and fully recomputed. This ensures tokenizer behavior is identical to the reuse-based runs, providing a fair accuracy upper-bound comparison.
```bash
# --- Greedy Decoding ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_full_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 1000000

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_full_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 1000000

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_full_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 1000000

# --- Sampling (seed=42) ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_full_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 1000000 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_full_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 1000000 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_full_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 1000000 \
    --do-sample --seed 42 --temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.1
```

#### Direct Reuse (RoPE Correction Only, No Rectification)
Direct Reuse applies only RoPE position correction to the reused decoding KV caches, without any selective rectification. We use EPIC with `epic_prefix_token_num=0`, meaning zero tokens are recomputed as prefix — all KV caches are directly reused after RoPE correction. This serves as the lower-bound baseline showing the quality degradation when no rectification is applied.
```bash
# --- Greedy Decoding ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_zero_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 0

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_zero_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 0

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_zero_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 0

# --- Sampling (seed=42) ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_zero_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 0 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_zero_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 0 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_zero_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 0 \
    --do-sample --seed 42 --temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.1
```

#### EPIC (16 prefix tokens)
EPIC recomputes the first 16 tokens as prefix before reusing the rest. This is the strongest prefix-caching baseline for multi-agent systems.
```bash
# --- Greedy Decoding ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_epic_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 16

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_epic_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 16

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_epic_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 16

# --- Sampling (seed=42) ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_epic_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 16 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_epic_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 16 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_epic_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode epic --epic_prefix_token_num 16 \
    --do-sample --seed 42 --temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.1
```

#### CacheBlend
CacheBlend recomputes KV caches for tokens where the hidden-state difference across layers exceeds a threshold (`cacheblend_diff_por=0.2`).
```bash
# --- Greedy Decoding ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_cacheblend_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode cacheblend --cacheblend_diff_por 0.2

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_cacheblend_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode cacheblend --cacheblend_diff_por 0.2

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_cacheblend_greedy \
    --cuda-devices 0 --batch_size 10 --execution_mode cacheblend --cacheblend_diff_por 0.2

# --- Sampling (seed=42) ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_cacheblend_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode cacheblend --cacheblend_diff_por 0.2 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_cacheblend_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode cacheblend --cacheblend_diff_por 0.2 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_cacheblend_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 10 --execution_mode cacheblend --cacheblend_diff_por 0.2 \
    --do-sample --seed 42 --temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.1
```

#### KVCOMM
KVCOMM uses KV-cache communication between rounds, reusing caches via anchor tokens and windowed KV collection (`kv-threshold=0.3`, `kv-max-anchor-num=20`).

> **Note:** Use `--batch_size 1` for KVCOMM main experiments. Samples are processed sequentially so anchor matching and KV reuse align across the run; larger batch sizes change processing order and typically yield a lower measured reuse rate.
```bash
# --- Greedy Decoding ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_kvcomm_greedy \
    --cuda-devices 0 --batch_size 1 --execution_mode kvcomm \
    --kv-threshold 0.3 --kv-max-anchor-num 20 --kv-window-size 5

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_kvcomm_greedy \
    --cuda-devices 0 --batch_size 1 --execution_mode kvcomm \
    --kv-threshold 0.3 --kv-max-anchor-num 20 --kv-window-size 5

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_kvcomm_greedy \
    --cuda-devices 0 --batch_size 1 --execution_mode kvcomm \
    --kv-threshold 0.3 --kv-max-anchor-num 20 --kv-window-size 5

# --- Sampling (seed=42) ---
# MMLU
python experiments/main/run_mmlu.py --mode FullConnected --agent_names AnalyzeAgent --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/mmlu_kvcomm_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 1 --execution_mode kvcomm \
    --kv-threshold 0.3 --kv-max-anchor-num 20 --kv-window-size 5 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# GSM8K
python experiments/main/run_gsm8k.py --mode FullConnected --agent_names MathSolver --agent_nums 3 \
    --llm_name meta-llama/Llama-3.1-8B-Instruct --output_dir ./runs/gsm8k_kvcomm_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 1 --execution_mode kvcomm \
    --kv-threshold 0.3 --kv-max-anchor-num 20 --kv-window-size 5 \
    --do-sample --seed 42 --temperature 0.6 --top-p 0.9

# HumanEval
python experiments/main/run_humaneval.py --mode FullConnected --agent_names CodeWriting --agent_nums 3 \
    --llm_name Qwen/Qwen2.5-Coder-7B-Instruct --output_dir ./runs/humaneval_kvcomm_nogreedy_seed42 \
    --cuda-devices 0 --batch_size 1 --execution_mode kvcomm \
    --kv-threshold 0.3 --kv-max-anchor-num 20 --kv-window-size 5 \
    --do-sample --seed 42 --temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.1
```



</details>

<details>
<summary><b>Observation Experiments</b></summary>

KV alignment between decoding and full-prefill (`run_observation.py`). Writes plots to `observation_results/plots/` and raw data to `observation_results/data/observation_data.json`. Layer auto-selection flags match the profiler (see `run_observation.py --help`).

```bash
python experiments/observation/run_observation.py --mode run \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --data ./datasets_self/wikimqa_s.json \
    --dataset-type wikimqa \
    --st-idx 0 --data-len 1 --ctx-num 1 \
    --output ./observation_results
```

</details>

<details>
<summary><b>Layer-Range Profiler</b></summary>

### Layer auto-selection (profiler CLI)

Same algorithm as observation plotting; profiler exposes shorter flag names:

| Profiler flag | Observation flag | Default |
|---------------|------------------|---------|
| `--tau-low` | `--layer-lower-threshold` | `0.99` |
| `--stable-multiplier` | `--layer-upper-sigma-multiplier` | `0.5` |
| `--consecutive-stables` | `--layer-upper-consecutive-count` | `2` |
| `--min-rise-samples` | `--layer-upper-min-samples` | `3` |
| `--right-tail-layers` | `--right-tail-layers` | `5` |

Override manually with `--start-layer`, `--det-layer`, and `--end-layer` together (disables auto-select).

### Running the Profiler

```bash
python experiments/profiler/run_layer_profiler.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data ./datasets_self/wikimqa_s.json \
    --dataset-type wikimqa \
    --st-idx 0 --data-len 1 --ctx-num 1 --device cuda:0
```

</details>

## Implementation Lessons

While implementing RelayCaching and the baselines in this codebase, we found that low-level setup choices—tokenization, cache construction, attention backends, dtype, and hardware—can confound what you measure (accuracy, reuse rate, latency) for methods that read or write KV caches, hidden states, and attention. The effect can look like a property of the method itself when you benchmark multiple approaches or re-run the same scripts on another GPU stack.

Below is what we aligned in the main experiments in this repo. Details are in each lesson.

| # | Align | Why it matters |
|---|--------|----------------|
| 1 | Tokenization path | Segment-wise vs. full-prompt tokenization can change token ids—especially when prompt or cache-reuse construction differs. |
| 2 | Attention backend | SDPA for all baselines in our runs; recompute attention scores when needed instead of switching backends for one baseline only. |
| 3 | Numerical precision | Same dtype everywhere (e.g. `torch.bfloat16`) for KV, RoPE, hidden states, and attention signals. |
| 4 | GPU hardware | Same GPU generation for all baselines; small shared-machine sanity check before large sweeps if jobs move across machines. |

<details>
<summary>Lesson 1: Align the tokenization path</summary>

Tokenizing prompt segments separately and then concatenating token ids can differ from tokenizing the full prompt once. That shows up quickly when baselines use different prompt construction, prefix caching, or cache-reuse layouts.

Use the same tokenization and generation path across baselines when you can. If a baseline changes how the prompt is built, check that token ids match the reference path before trusting metric deltas.

</details>

<details>
<summary>Lesson 2: Align the attention backend</summary>

Attention backends differ in numerics and in which intermediates they expose. Changing backend for only one baseline can confound benchmarks.

For accuracy and reuse-rate experiments here, we keep backends aligned and use SDPA for all baselines. That way backend effects are less likely to be read as method behavior.

Eager attention exposes attention weights directly (useful for token-level influence). SDPA/FlashAttention-style kernels often do not. When we need scores, we recompute them under the same SDPA setting rather than switching to eager for one baseline only. Main experiments avoid FlashAttention-only paths for the same reason.

</details>

<details>
<summary>Lesson 3: Align numerical precision</summary>

Keep one model precision (e.g. `torch.bfloat16`) across baselines. KV caches, RoPE-dependent tensors, hidden states, and attention signals are sensitive to dtype; mixed precision can shift metrics without changing the intended method logic.

</details>

<details>
<summary>Lesson 4: Use consistent GPU hardware</summary>

Different GPU architectures can change floating-point outcomes slightly. Token-level decisions driven by small numerical signals (as in selective KV rectification) are especially sensitive.

Run baselines on the same GPU type. If some runs move to another machine, repeat a small subset on one shared host before trusting full-scale numbers.

</details>

<details>
<summary>General note (library versions, determinism, and other confounds)</summary>

This list is not complete—library versions and CUDA non-determinism matter too. Treat any unaligned low-level detail (across baselines or across machines) as a possible confound; align what you can and validate on a small run before large evaluations.

</details>


## Citation

```bibtex
@misc{geng2026relaycachingacceleratingllmcollaboration,
      title={RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse}, 
      author={Yingsheng Geng and Yuchong Gao and Weihong Wu and Guyue Liu and Jiang Liu},
      year={2026},
      eprint={2603.13289},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.13289}, 
}
```

## Acknowledgements

This project is built upon [KVCOMM](https://github.com/FastMAS/KVCOMM). We thank the authors for open-sourcing their codebase.


## License

This project is released under the [MIT License](LICENSE).
