# Cross-prefix KV reuse dataset pipeline

This repository currently implements pinned downloads and deterministic cross-prefix construction for IBM ArgKP, NVIDIA HelpSteer2, Facebook's Deal or No Deal negotiation corpus, PKU-SafeRLHF, and HarmBench contextual behaviors.

## Quick start

Python 3.10+ is sufficient; the data pipeline uses only the standard library.

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/download_datasets.py
python scripts/build_datasets.py --seed 20260827 --max-per-dataset 1000
python scripts/validate_datasets.py
```

Raw downloads are cached under `data/raw/`. Every file's URL, pinned source revision, size, and SHA-256 digest are recorded in `data/raw/manifest.json`. Existing non-empty files are reused unless `--force` is supplied. Constructed files are stored under `data/processed/`, with five human-checkable examples per dataset in `data/processed/examples/`.

Use `--max-per-dataset 0` to retain all valid examples. Sampling is deterministic for a fixed seed. `data_config.json` documents the default experiment values; command-line flags are authoritative.

## Environments

The dataset-only pipeline above needs no third-party Python packages. Model inference
and analysis use a CUDA-capable GPU and two isolated environments: `kvreuse` for this
repository's Qwen3 runners, and `relaycaching` for the external baselines. Do not
install the RelayCaching requirements into `kvreuse`. Choose either the Conda or the
venv recipe below; both create the same two environments.

### Conda installation

```bash
# Main project: direct reuse, Tail-16, bridge, KVCOMM, and analysis.
conda create -y -n kvreuse python=3.10
conda run -n kvreuse python -m pip install --upgrade pip

# Install a CUDA-enabled PyTorch wheel appropriate for your NVIDIA driver first.
# See https://pytorch.org/get-started/locally/ for the exact command.
conda run -n kvreuse python -m pip install torch transformers modelscope numpy matplotlib

# External methods: RelayCaching, CacheBlend, and EPIC.
git clone https://github.com/YingshengGeng/RelayCaching.git third_party/RelayCaching
git -C third_party/RelayCaching checkout 6b0e9d37b74d89cfb0fbad276790150a0d9c0b49
conda create -y -n relaycaching python=3.12
conda run -n relaycaching python -m pip install --upgrade pip
conda run -n relaycaching python -m pip install -r third_party/RelayCaching/requirements.txt
```

### venv installation

```bash
# Main project environment.
python3.10 -m venv .venv_kvreuse
. .venv_kvreuse/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-enabled PyTorch wheel appropriate for your NVIDIA driver first.
python -m pip install torch transformers modelscope numpy matplotlib
deactivate

# External-baseline environment (the upstream project uses Python 3.12).
git clone https://github.com/YingshengGeng/RelayCaching.git third_party/RelayCaching
git -C third_party/RelayCaching checkout 6b0e9d37b74d89cfb0fbad276790150a0d9c0b49
python3.12 -m venv .venv_relaycaching
. .venv_relaycaching/bin/activate
python -m pip install --upgrade pip
python -m pip install -r third_party/RelayCaching/requirements.txt
deactivate
```

Verify that the main environment sees CUDA before inference:

```bash
conda run -n kvreuse python -c \
  "import torch, transformers, modelscope; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, modelscope.__version__)"
```

The development environment used for the reported runs was Python 3.10.20, PyTorch
2.13.0+cu130, Transformers 5.15.0, ModelScope 1.39.1, NumPy 2.2.6, and Matplotlib
3.10.9. `torch.cuda.is_available()` must print `True` before running inference.

### Downloading models to a non-home filesystem

All Qwen3 runners honor `MODELSCOPE_CACHE`. Set it to a directory with enough space
before downloading and keep the same variable exported while running benchmarks. For
example, to store checkpoints under `/data` rather than `~/.cache/modelscope`:

```bash
export MODELSCOPE_CACHE=/data/$USER/modelscope
mkdir -p "$MODELSCOPE_CACHE"

# Download only the sizes needed for an experiment; run on a network-enabled node.
conda run -n kvreuse python scripts/download_models.py \
  --cache-dir "$MODELSCOPE_CACHE" --models 1.7b 4b 8b

# Confirm the resolved cache locations and that no checkpoint shard is missing.
conda run -n kvreuse python scripts/download_models.py \
  --cache-dir "$MODELSCOPE_CACHE" --models 4b
```

ModelScope stores Qwen3-4B at
`$MODELSCOPE_CACHE/models/Qwen--Qwen3-4B/snapshots/<revision>`. The direct runners,
KVCOMM, and the RelayCaching/CacheBlend/EPIC adapter all use this environment variable.
For a detached shell or tmux job, export it in the command itself:

```bash
tmux new-session -d -s kvreuse_4b \
  'cd /path/to/kvreuse && MODELSCOPE_CACHE=/data/$USER/modelscope \
   bash scripts/run_benchmark_301.sh --model-size 4b --reasoning off'
```

### Clean-machine `/data` layout (venv example)

After cloning this repository and following the venv installation recipe, the source
checkout, environments, datasets, models, and outputs may all live outside `$HOME`.
The following end-to-end sequence uses `/data/$USER/kvreuse` for every large artifact:

```bash
git clone <this-repository-url> /data/$USER/kvreuse/source
cd /data/$USER/kvreuse/source

# Create .venv_kvreuse and .venv_relaycaching here using the venv recipe above.
export DATA_ROOT=/data/$USER/kvreuse/data
export MODELSCOPE_CACHE=/data/$USER/kvreuse/modelscope
export RESULT_ROOT=/data/$USER/kvreuse/results/benchmark_301
mkdir -p "$DATA_ROOT" "$MODELSCOPE_CACHE" "$RESULT_ROOT"

# Download and construct data entirely under /data.
. .venv_kvreuse/bin/activate
python scripts/download_datasets.py --output-dir "$DATA_ROOT/raw"
python scripts/build_datasets.py --raw-dir "$DATA_ROOT/raw" --output-dir "$DATA_ROOT/processed" \
  --seed 20260827 --max-per-dataset 1000
python scripts/validate_datasets.py "$DATA_ROOT/processed/all.jsonl"
python scripts/prepare_benchmark.py --input "$DATA_ROOT/processed/all.jsonl" \
  --datasets argkp deal_or_no_deal harmbench_contextual --per-dataset 110 --allow-fewer \
  --output "$DATA_ROOT/benchmark/benchmark_argkp_deal_harmbench_301.jsonl"

# Download the desired checkpoint into /data, then run the suite.
python scripts/download_models.py --cache-dir "$MODELSCOPE_CACHE" --models 8b
bash scripts/run_benchmark_301.sh --model-size 8b --reasoning off \
  --input "$DATA_ROOT/benchmark/benchmark_argkp_deal_harmbench_301.jsonl" \
  --output-root "$RESULT_ROOT" \
  --relay-python "$PWD/.venv_relaycaching/bin/python"
```

The suite script uses the active `python` by default. If it is not activated, set
`KVREUSE_PYTHON=/data/$USER/kvreuse/source/.venv_kvreuse/bin/python` before invoking
the script. `--input` and `--output-root` make the benchmark independent of the
repository's `data/` and `results/` directories.

## Construction and filtering

All records contain `task_id`, `dataset`, `prefix_a`, `prefix_b`, `shared_block`, `question`, `gold_a`, `gold_b`, and `metric`. Additional `metadata` preserves source IDs and construction facts for auditing. The validator rejects empty required fields and every record for which `gold_a == gold_b`.

- **ArgKP:** each block has six arguments about one topic. The PRO prefix supplies a pro key point and the CON prefix supplies a con key point. Each target has exactly one positively annotated candidate; same-stance distractors have explicit negative labels. Samples lacking two stances, a unique match, or enough annotated negatives are excluded.
- **HelpSteer2:** the exact same prompt-response block is scored under two of correctness, helpfulness, coherence, complexity, and verbosity. Prefixes include the official paper's dimension-specific 0–4 anchors. Complexity is a basic-to-expert language spectrum and verbosity is a succinct-to-verbose spectrum; neither is a generic bad-to-good score. All ten rubric pairs are considered. To avoid unstable middle-score calibration, the public per-annotator data is used to retain only pairs where every annotator agrees and the two dimensions have opposite endpoint labels: one is exactly 0 and the other exactly 4. The model therefore chooses only between `0` and `4`; intermediate labels are excluded.
- **Deal or No Deal:** item counts and both agents' utilities come from the corpus. The block shows the same six feasible allocations in compact `(books, hats, balls)` tuples. Prefixes reveal Agent A's or Agent B's utility function. Candidate sets with tied optima or the same optimum for both agents are rejected. Gold is computed directly from the linear utility functions.
- **PKU-SafeRLHF:** the block contains one user prompt and the same two candidate responses. Prefix A selects only by the dataset's helpfulness preference; prefix B selects only by safety. The high-confidence pool requires `better_response_id != safer_response_id`, exactly one safe response, the safety preference pointing to that response, safe severity 0, unsafe severity at least 2, no harm category on the safe response, and at least one harm category on the unsafe response. Duplicate prompt/response pairs are removed. Response A/B order is deterministically randomized, and the retained pool is stratified across source model, primary harm category, unsafe severity, and displayed helpfulness answer. Only the three official test shards are downloaded (~23 MB), not the full training repository.
- **HarmBench contextual:** only official test rows with `FunctionalCategory=contextual`, a `context` tag, and a non-empty `ContextString` are retained. The exact `ContextString` is the shared block. Prefix A describes a defensive audit/mitigation purpose that will not operationalize the material; prefix B contains the official harmful `Behavior` and describes applying the material to carry it out. The automatically scored output flips from `BENIGN` to `HARMFUL`. This is an intent-classification task and never asks the evaluated model to generate the harmful behavior. Validation sampling diversifies semantic harm category and context-length band.

These filters establish a machine-checkable change in the correct output. They do not prove that every natural-language annotation is free from subjective ambiguity; the example files are provided for manual review before model experiments.

## Data sources

- ArgKP: official `IBM/KPA_2021_shared_task` repository, pinned in the downloader.
- HelpSteer2: `nvidia/HelpSteer2`, pinned by Hugging Face revision. `HF_ENDPOINT` can select the official endpoint or a mirror.
- Deal or No Deal: official `facebookresearch/end-to-end-negotiator` repository, pinned in the downloader.
- PKU-SafeRLHF: official `PKU-Alignment/PKU-SafeRLHF` test shards for Alpaca-7B, Alpaca2-7B, and Alpaca3-8B, pinned to revision `9421ffafec3fa40a1f1a7d567b4d525079477ecb`. License: CC BY-NC 4.0.
- HarmBench: official `centerforaisafety/HarmBench` text test behavior CSV, pinned to revision `8e1604d1171fe8a48d8febecd22f600e462bdcdd`. License: MIT. The source contains harmful material and is intended only for controlled safety research.

Respect the licenses and dataset cards copied into each raw dataset directory.

## Scope

In addition to dataset construction, this repository now contains Qwen3 full-prefill,
direct KV reuse, clean-block reuse, Tail-16 recomputation, target-task bridge ablations,
KVCOMM-style calibration, and adapters for the RelayCaching, CacheBlend, and EPIC
baselines. Experiment outputs are deliberately kept under `results/` and are not source
files; benchmark definitions and runners are versioned.

## Direct-reuse development runner

The uncommitted next-milestone runner evaluates each selected record atomically as
`full_A`, `full_B`, `A->B`, and `B->A`:

```bash
conda run -n kvreuse python scripts/run_direct_reuse.py \
  --model 0.6b \
  --input data/validation/second_step_10_each_harmbench.jsonl \
  --max-samples 10 \
  --output-root results/direct_reuse
```

Add `--run-self-controls` after a model, Transformers, attention backend, or cache
implementation change. It additionally runs `A->A` and `B->B` through the identical
extraction, relocation, splice, suffix, and generation path as cross-prefix reuse.
Once those controls pass, formal runs omit them to avoid duplicating `full_A/full_B`.
Post-RoPE cached Keys are inverse-rotated at their source absolute positions and
rotated into the target positions through Qwen3's own rotary embedding; Values are
copied without rotation. The runner rejects non-default RoPE configurations and any
record whose token-aligned shared block differs between A and B.

Every completed sample is appended immediately to `samples.jsonl`. Its artifact
directory contains compressed token-by-layer cosine matrices and a two-row Key/Value
heatmap whose x-axis is decoder layer and y-axis is shared-block token. Self-control
runs produce four rows instead. The NPZ also
retains uncorrected-Key cosine so RoPE relocation can be audited directly. `summary.json`
is refreshed after every sample with overall and per-dataset full-vs-reuse accuracy
loss, prediction agreement, first-token KL, latency, and KV similarity.

Reuse latency includes target-prefix prefill, RoPE relocation/cache splice, suffix
prefill, and greedy generation. It excludes creation of the donor shared-block cache,
which is treated as already resident. BF16/SDPA segmented prefill is not bitwise equal
to a monolithic full prefill, so self controls require identical generated token IDs,
first-token KL at most `0.01`, and logit cosine at least `0.999`; maximum absolute logit
difference remains recorded as a diagnostic rather than a pass criterion.

## Ours: bridge reuse ablation

`scripts/run_ours_bridge_reuse.py` is a separate method entry point; it does not
change the original `full`, `reuse`, or `clean_reuse` prompt construction.

```bash
conda run -n kvreuse python scripts/run_ours_bridge_reuse.py \
  --bridge post_task_restatement --model 0.6b \
  --input data/benchmark/benchmark_250.jsonl \
  --method all --boxed-output \
  --output-root results/benchmark_250/no_reasoning/ours_post
```

`post_task_restatement` adds a target-role restatement after the transplanted
block and before the question.  Its donor block KV is unchanged by causality.
`pre_block_precaution` adds the caution immediately before the block; this is a
separate precaution-conditioned baseline, so both donor and target block KVs see
the warning.  Run each variant under a different output root.

## External baseline: RelayCaching, CacheBlend, and EPIC

The three external baselines are run through `scripts/run_relay_methods.py`, which
adapts this repository's static shared-block task into the official RelayCaching
code path. The external repository is intentionally not vendored in Git. Obtain it
under the expected path and install its requirements in a separate environment:

```bash
git clone https://github.com/YingshengGeng/RelayCaching.git third_party/RelayCaching
git -C third_party/RelayCaching checkout 6b0e9d37b74d89cfb0fbad276790150a0d9c0b49

conda create -n relaycaching python=3.12
conda run -n relaycaching python -m pip install -r third_party/RelayCaching/requirements.txt
```

The official upstream installation uses `uv`; the separate Conda or venv environment
above is equivalent for this repository's runners. The default launcher expects
`/home/czw/miniconda3/envs/relaycaching/bin/python`; for venv or another Conda prefix,
supply the environment's interpreter with `--relay-python`.

Run one external baseline through the common benchmark launcher:

```bash
python scripts/run_benchmark.py --method relaycaching --reasoning no --model 4b \
  --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
  --output-root results/benchmark_argkp_deal_harmbench_301 \
  --relay-python /path/to/relaycaching/bin/python
```

Replace `relaycaching` with `cacheblend` or `epic` to run the other two baselines.
`scripts/run_relay_methods_all.sh 4b` is a convenience wrapper when invoking the
external methods directly. Their sample-level files use one row per cross direction;
the direct-KV runners use one row containing both directions.

## KVCOMM-style calibration

`scripts/run_kvcomm.py` implements the prefix-offset KVCOMM-style calibration used
in this project. It does not require external source code, but it requires an existing
Full run with the identical model, input JSONL, and reasoning mode. The unified launcher
enforces that dependency:

```bash
python scripts/run_benchmark.py --method full --reasoning no --model 4b \
  --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
  --output-root results/benchmark_argkp_deal_harmbench_301

python scripts/run_benchmark.py --method kvcomm --reasoning no --model 4b \
  --input data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl \
  --output-root results/benchmark_argkp_deal_harmbench_301
```

With `--method all`, Full runs before KVCOMM automatically.

## Building another benchmark slice

The benchmark builder accepts an explicit dataset list and common quota.  For
example, the 301-record ArgKP/Deal/HarmBench slice (110 / 110 / 81; HarmBench
uses all available records) is built with:

```bash
python scripts/prepare_benchmark.py \
  --datasets argkp deal_or_no_deal harmbench_contextual \
  --per-dataset 110 --allow-fewer \
  --output data/benchmark/benchmark_argkp_deal_harmbench_301.jsonl
```

Run the ten-method suite (`Full`, `Reuse`, `Clean`, `Tail-16`, `Tail-16 + post`,
`Post`, `RelayCaching`, `CacheBlend`, `EPIC`, and `KVCOMM`) against any newly
generated JSONL:

```bash
bash scripts/run_benchmark_301.sh --model-size 4b --reasoning off
```

For a custom input or a single method, use `scripts/run_benchmark.py` directly.
The suite runner accepts `--model-size {1.7b,4b,8b}` and `--reasoning {on,off}`,
resumes completed sample files by default, and writes a method-by-dataset table under
`analysis/outputs/` after all methods finish.

### HelpSteer2 + PKU-SafeRLHF evaluation

`data/benchmark/benchmark_helpsteer2_pku_safe_rlhf_266.jsonl` contains a deterministic
balanced evaluation set: 133 HelpSteer2 records and 133 PKU-SafeRLHF records. Its
manifest records the source checksum, seed, counts, and selected task IDs. Run the
following command to rebuild the dataset from `data/processed/all.jsonl`:

```bash
python scripts/prepare_benchmark.py --preset helpsteer2_pku_safe_rlhf_266 --overwrite
```

Run the default Full baseline with:

```bash
bash scripts/run_helpsteer2_pku_safe_rlhf_266.sh --model-size 4b --reasoning off
```

The runner also accepts `--method` (for example `reuse` or `all`), `--limit` for a
smoke test, and `--output-root`; model sizes are `0.6b`, `1.7b`, `4b`, and `8b`.
