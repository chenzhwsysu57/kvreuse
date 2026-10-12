# Latent Remedy: prefix-conditioned KV cache repair

A clean release of Qwen3 cross-prefix KV reuse, prefix-conditioned latent-adapter
training, inference, and selected evaluation data. Model weights, adapter checkpoints,
experiment outputs, private credentials, cluster launchers and historical Git history
are not included.

The Python package `latent_remedy` contains the prefix-conditioned KV cache repair
training and inference implementation. Use `latent_remedy` in Python imports and
package paths; `latent-remedy` is only a project display name, not a Python package.

## Branches

- **main**: training, adapter evaluation, full-prefill/direct reuse/clean reuse,
  Repeat-prefix, bridge, KVCOMM, universal soft suffix, and serial RelayCaching,
  CacheBlend and EPIC runners. Existing first-party batching is retained.
- **batch**: main plus the RelayCaching batching cache/Qwen3 changes and runner
  support. `official-serial` remains the default; `static-batched` is a diagnostic
  throughput path and must not be presented as official paper TTFT results.

## Environment and models

Use Python 3.10+ and a CUDA GPU for real model training/inference. Create a Conda
environment, install a CUDA-enabled PyTorch wheel appropriate for the host driver,
then install [requirements.txt](requirements.txt). The direct-dependency pins
record the source environment (Python 3.10.20, PyTorch 2.13.0+cu130,
Transformers 5.15.0). A clean installation and real GPU runs are not claimed to have
been repeated as part of repository packaging.

```bash
conda create -n kvreuse-public python=3.10 -y
conda activate kvreuse-public
python -m pip install -r requirements.txt
export MODELSCOPE_CACHE=/path/to/model-cache
python scripts/download_models.py --models 4b --cache-dir "$MODELSCOPE_CACHE"
```

The download script reports the local model directory. For adapter training and
evaluation, set `MODEL_PATH` to that directory. Runners default to the ModelScope
`models/Qwen--Qwen3-4B/snapshots/master` layout; if using a different downloaded
revision, pass the reported local directory with `--model-path` to the direct or
adapter entry points. The unified [run.py](run.py) does not expose `--model-path`;
its full/reuse and Relay commands use the expected layout under `MODELSCOPE_CACHE`.
Model weights and trained adapters are not shipped.

### External baselines

RelayCaching uses a different Transformers version and requires a separate Python
3.12 environment. Its source is vendored; no Git submodule initialization is needed.
FlashAttention and vLLM are not required for the default Hugging Face/SDPA runner.

```bash
conda create -n kvreuse-public-relay python=3.12 -y
conda run -n kvreuse-public-relay python -m pip install -r requirements-relay.txt
# Replace this with the absolute Python executable in the Relay environment.
export RELAY_PYTHON=/absolute/path/to/envs/kvreuse-public-relay/bin/python
```

## Data

[data/README.md](data/README.md) describes the only shipped data:

| Dataset alias | Paired records | Scope |
|---|---:|---|
| `argkp` | 110 | ArgKP real-world evaluation |
| `deal` | 110 | Deal or No Deal evaluation |
| `job` | 110 | Job Interview evaluation |
| `all330` | 330 | The three real-world sets combined |
| `all240` | 7,200 | 240 synthetic task families, 30 pairs each |
| `train180` | 5,400 | 180 training families, 30 pairs each |
| `test60` | 1,800 | 60 disjoint held-out families, 30 pairs each |

The generated data is the latest complete 240-family corpus with ten Deal/Job-style
scene families replaced (2026-10-08), not an old adapter fine-tuning corpus. Training
data is retained because it is necessary to reproduce adapter training. Historical
synthetic suites and unrelated real datasets/raw downloads are omitted. A pair can
be evaluated in two directions; paired-record counts are not directional counts.

Each shipped record contains seven core fields: `dataset`, `prefix_a`, `prefix_b`,
`shared_block`, `question`, `gold_a`, and `gold_b`. Optional `output_protocol` is
retained only for non-`option_letter` protocols; omission means `option_letter`.
`task_id` is derived at runtime and `metric` defaults to `exact_match`; neither is
stored. Metadata, compiler/seed/oracle/audit fields and all other extra fields are
removed. Real-world `dataset` values remain `argkp`, `deal_or_no_deal`, and
`job_interview`; synthetic names are consistently anonymous `family_001` through
`family_240` across full/train/test.

**Prompts, gold answers, family groupings and the family-level 180/60 split are
unchanged; JSONL bytes and runtime IDs are not preserved.** New runtime IDs cannot
resume old outputs or reuse old ID-keyed caches. Use fresh results directories,
regenerate teacher/state caches and direction-ID lists, and do not bypass the
incompatibility with `--overwrite`. There is no data manifest; verification does
not depend on one. The data verifier checks counts, minimal fields and split membership.

## Full-prefill and direct reuse with run.py

```bash
python run.py --dataset all330 --method full --model 4b \
  --reasoning no --kv-mode all --batch-size 4 \
  --output-root results/compact_v1/full_no
python run.py --dataset test60 --method reuse --model 4b \
  --reasoning yes --kv-mode all --batch-size 4 \
  --output-root results/compact_v1/reuse_yes
```

`--dry-run` checks selection and writes the planned command without loading a model.
`--input` accepts another compatible paired JSONL instead of `--dataset`; use
`--input-label` for a safe output label. Results default to an ignored output
directory. `--overwrite` explicitly allows replacing an existing run directory.
KVCOMM additionally requires `--baseline` pointing to full-prefill sample outputs;
universal soft suffix requires `--soft-prompt-checkpoint`.

**Prompt modes are not the Qwen thinking switch.** Both `--reasoning no` and
`--reasoning yes` keep `enable_thinking=False`; the latter requests explicit
reasoning through the prompt (`explicit_reasoning=True`).

## Train from scratch: train180, three pairs per family

The authoritative implementation is
[latent_remedy/train_prefix_adapter_repeat.py](latent_remedy/train_prefix_adapter_repeat.py).
The root training entry point is a small compatibility wrapper. A minimal from-scratch
example is below; it is not a claim of exact paper hyperparameter reproduction.
It uses no-reasoning prompts and generated full-prefill teacher traces, not the
exact mixed-supervision paper configuration. No old adapter is loaded.

```bash
export MODEL_PATH=/path/to/local/Qwen3-4B
python train_prefix_adapter_repeat.py \
  --input data/benchmark/generated240/train_180classes_30each.jsonl \
  --model 4b --model-path "$MODEL_PATH" --output results/compact_v1/adapter_train180 \
  --train-records-per-dataset 3 --train-reasoning no \
  --teacher-mode full --teacher-trace generated \
  --latent-tokens 16 --adapter-bottleneck 256 --epochs 1 --train-batch-size 4
```

`--train-records-per-dataset 3` selects three source pairs per family (540 pairs
for train180), then trains both A/B directions. **Do not add `--use-all-records`:**
it bypasses three-pair sampling and uses all 30 pairs per family. Keep the remaining
pairs and test60 families out of training and checkpoint selection.

### Change the input for all240

Replace the training input with
[data/benchmark/generated240/full_240classes_30each.jsonl](data/benchmark/generated240/full_240classes_30each.jsonl)
and choose a fresh output such as `results/compact_v1/adapter_all240`. Keep
`--train-records-per-dataset 3`: all240 means all 240 families (720 selected pairs),
not all 30 pairs per family. Test60 is not held out from all240 training; evaluate
that adapter on the separate real-world all330 input for OOD results.

## Evaluate an adapter: test60 and all330

```bash
python latent_remedy/eval_prefix_adapter_repeat.py \
  --checkpoint results/compact_v1/adapter_train180/latest_prefix_adapter.pt \
  --input data/benchmark/generated240/test_60classes_30each.jsonl \
  --model-path "$MODEL_PATH" --output results/compact_v1/adapter_test60_no \
  --reasoning no --max-new-tokens 128 --batch-size 4
python latent_remedy/eval_prefix_adapter_repeat.py \
  --checkpoint results/compact_v1/adapter_train180/latest_prefix_adapter.pt \
  --input data/benchmark/benchmark_final240_real_ood_330_20261006.jsonl \
  --model-path "$MODEL_PATH" --output results/compact_v1/adapter_all330_no \
  --reasoning no --max-new-tokens 128 --batch-size 4
```

For the all240-trained adapter, replace the checkpoint and choose another fresh
all330 output directory. For explicit reasoning, use `--reasoning yes
--max-new-tokens 4096` with a separate output directory; native thinking stays off.
Universal soft-suffix training/evaluation is also available through
[train-latent-token.py](train-latent-token.py) and
[test-latent-token.py](test-latent-token.py).

## RelayCaching: main serial and batch static-batched

This same README describes both branches. Launch [run.py](run.py) in the main
environment and explicitly pass the separate Relay interpreter.

### main: official serial path

```bash
python run.py --dataset all330 --method relaycaching --model 4b \
  --reasoning no --kv-mode all --batch-size 1 \
  --relay-python "$RELAY_PYTHON" \
  --output-root results/compact_v1/relay_main_serial
```

Main runs Relay serially. **Do not pass `--relay-execution-path` or
`--warmup-batches` on main:** these batch-only flags are absent from main's parser.
First-party batch support does not imply Relay batching support.

### batch only: static-batched diagnostic path

Switch branches only with a clean working tree.

```bash
git switch batch
python run.py --dataset all330 --method relaycaching --model 4b \
  --reasoning no --kv-mode all --relay-python "$RELAY_PYTHON" \
  --relay-execution-path static-batched --batch-size 2 --warmup-batches 1 \
  --output-root results/compact_v1/relay_batch_static_b2
```

Keep serial and batch output roots separate. `official-serial` remains the batch
default; `static-batched` is diagnostic throughput, not official paper TTFT or proof
of numerical equivalence. Both prompt modes keep `enable_thinking=False`;
`--reasoning yes` changes only explicit prompt reasoning. The batch parser must be
used on the batch branch; those flags are unavailable on main.

## CPU checks in two environments; reproduction limits

Run first-party CPU checks in the main environment from the repository root. The
verifier checks compact structure and membership without a manifest, not historical
hashes or equality to pre-cleanup prompts.

```bash
conda activate kvreuse-public
python scripts/verify_public_data.py
python -m unittest discover -s tests -v
python -m compileall -q run.py scripts latent_remedy kvreuse_data
python train_prefix_adapter_repeat.py --help
python latent_remedy/eval_prefix_adapter_repeat.py --help
python run.py --help
```

Use the Relay environment for its runner help and, on batch, Relay-specific CLI/cache
tests. Transformers 5.x skips in main discovery do not replace these checks.

```bash
"$RELAY_PYTHON" scripts/run_relay_methods.py --help
# On batch only; run from the repository root.
"$RELAY_PYTHON" -m unittest discover -s tests -p 'test_relay*.py' -v
```

Tests use small CPU tensors/mocks; they do not establish full Qwen GPU accuracy,
throughput, or TTFT equivalence. See [docs/RELEASE_CHECKS.md](docs/RELEASE_CHECKS.md)
for actual packaging checks and their limits. No fresh installation or full-model
GPU reproduction was performed during packaging.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for licensing and provenance before public redistribution.