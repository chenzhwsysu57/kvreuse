# KV Reuse: prefix-conditioned latent repair

A clean release of Qwen3 cross-prefix KV reuse, prefix-conditioned latent-adapter
training, inference, and selected evaluation data. Model weights, adapter checkpoints,
experiment outputs, private credentials, cluster launchers and historical Git history
are not included.

## Branches

- **main**: training, adapter evaluation, full-prefill/direct reuse/clean reuse,
  Repeat-prefix, bridge, KVCOMM, universal soft suffix, and serial RelayCaching,
  CacheBlend and EPIC runners. Existing first-party batching is retained.
- **batch**: main plus the RelayCaching batching cache/Qwen3 changes and runner
  support. `official-serial` remains the default; `static-batched` is a diagnostic
  throughput path and must not be presented as official paper TTFT results.

## Installation

Use Python 3.10+ and a CUDA GPU for real model training/inference. Create a Conda
environment, install a CUDA-enabled PyTorch wheel appropriate for the host driver,
then install [requirements.txt](requirements.txt). The exact direct-dependency pins
match the source's verified environment (Python 3.10.20, PyTorch 2.13.0+cu130,
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
adapter entry points.

### External baselines

RelayCaching uses a different Transformers version and requires a separate Python
3.12 environment. Its source is vendored; no Git submodule initialization is needed.
FlashAttention and vLLM are not required for the default Hugging Face/SDPA runner.

```bash
conda create -n kvreuse-public-relay python=3.12 -y
conda run -n kvreuse-public-relay python -m pip install -r requirements-relay.txt
export RELAY_PYTHON="$(conda run -n kvreuse-public-relay python -c 'import sys; print(sys.executable)')"
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

## Inference

```bash
python run.py --dataset all330 --method full --model 4b \
  --reasoning no --kv-mode all --batch-size 4
python run.py --dataset test60 --method reuse --model 4b \
  --reasoning yes --kv-mode all --batch-size 4
python run.py --dataset all330 --method epic --model 4b \
  --reasoning no --kv-mode all --relay-python "$RELAY_PYTHON"
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

## Adapter training and evaluation

The authoritative implementation is
[kv_semantic_attack/train_prefix_adapter_repeat.py](kv_semantic_attack/train_prefix_adapter_repeat.py).
The root training entry point is a small compatibility wrapper. A minimal from-scratch
example is below; it is not a claim of exact paper hyperparameter reproduction.

```bash
export MODEL_PATH=/path/to/local/Qwen3-4B
python train_prefix_adapter_repeat.py \
  --input data/benchmark/generated240/train_180classes_30each.jsonl \
  --model 4b --model-path "$MODEL_PATH" --output results/adapter_train180 \
  --use-all-records --train-reasoning no --teacher-mode full \
  --latent-tokens 16 --adapter-bottleneck 256 --epochs 1 --train-batch-size 4
python kv_semantic_attack/eval_prefix_adapter_repeat.py \
  --checkpoint results/adapter_train180/latest_prefix_adapter.pt \
  --input data/benchmark/generated240/test_60classes_30each.jsonl \
  --model-path "$MODEL_PATH" --output results/adapter_test60 \
  --reasoning no --max-new-tokens 128 --batch-size 4
```

For `all240`, use the full corpus instead of the train180 corpus. The trainer exposes
sampling and supervision options through `--help`; the 30-pair input pools do not
mean every run must train on all 30 pairs. No old adapter is loaded unless explicitly
requested. Universal soft-suffix training/evaluation is also available through
[train-latent-token.py](train-latent-token.py) and
[test-latent-token.py](test-latent-token.py).

## Local checks

```bash
python scripts/verify_public_data.py
python -m unittest discover -s tests -v
python -m compileall -q run.py scripts kv_semantic_attack kvreuse_data
```

Tests use small CPU tensors/mocks; they do not establish full Qwen GPU accuracy,
throughput, or TTFT equivalence. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for licensing and provenance before public redistribution.