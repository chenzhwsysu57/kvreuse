<div align="center">


<h3>Efficient Remedy for Non-Prefix KV Cache Reuse<br>in Multi-Agent Systems</h3>

<p>Anonymous review artifact · <a href="data/README.md">ConflictInstruct data</a></p>

</div>

<p align="center">
  <a href="assets/accuracy-ttft.png"><img src="assets/accuracy-ttft.png" alt="Accuracy and TTFT comparisons on Task Switch, Boolean Logic, Label Mapping, and Priority Check across Qwen3 model sizes and prompt modes" width="1100"></a>
</p>

<p align="center"><strong>Accuracy–TTFT balance.</strong> Ours: <strong>Latent n</strong>.</p>

---

## Run it the vibe-coding way (recommended)

Just clone this repo and ask your Copilot, Codex, or Claude to install the dependencies for you and run full prefill, direct reuse, Repeat-prefix, or the baselines under [third_party/RelayCaching](third_party/RelayCaching/).

For example:

> Read the README and set up a single conda environment named latent-remedy with Python 3.10. Install the Relay base first with pip install -r requirements-relay.txt, then our own additions with pip install -r requirements.txt; the main requirements include -r requirements-relay.txt. Download Qwen3-4B, then run full prefill, direct reuse, Repeat-prefix, and the RelayCaching, CacheBlend, and EPIC baselines on all330 in this same environment. Start with a dry run and two pairs, keep native thinking disabled, and use separate output directories.

Third-party code and data retain their original terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Run it the manual way

### 1. Install dependencies and download a model

Use one conda environment, `latent-remedy`, with Python 3.10 for training, evaluation, and all baselines. Install the Relay base first, then our own additions; the main requirements include `-r requirements-relay.txt`. GPU runs require a CUDA-capable GPU and a CUDA-enabled PyTorch wheel compatible with your driver.

```bash
# Download and extract this anonymous repository, then enter its root directory.
cd kvreuse
conda create -n latent-remedy python=3.10 -y
conda activate latent-remedy
python -m pip install -r requirements-relay.txt
python -m pip install -r requirements.txt

export MODELSCOPE_CACHE=/path/to/model-cache
python scripts/download_models.py --models 4b --cache-dir "$MODELSCOPE_CACHE"
export MODEL_PATH=/path/to/downloaded/Qwen3-4B
```

Set `MODEL_PATH` to the directory reported by the download script. Adapter entry points accept `--model-path`; [run.py](run.py) uses the ModelScope cache layout under `MODELSCOPE_CACHE`. Models and adapter checkpoints are not included.

RelayCaching, CacheBlend, and EPIC use the same environment. [run.py](run.py) defaults to `sys.executable` for Relay subprocesses; `--relay-python /path/to/python` is an optional interpreter override, not required for this setup. The required third-party source is included; no submodule setup is needed.

Keep Transformers pinned to 4.52.4 for compatibility with the vendored Relay Qwen3/cache implementation.

### 2. Choose a dataset

| Alias | Paired records | Dataset |
|---|---:|---|
| `argkp` / `deal` / `job` | 110 each | Real-world evaluation |
| `all330` | 330 | ArgKP, Deal, and Job combined |
| `all240` | 7,200 | All 240 synthetic families |
| `train180` | 5,400 | 180 training families |
| `test60` | 1,800 | 60 held-out families |

Each synthetic family has 30 pairs. See [data/README.md](data/README.md) for the input format and data terms. Use `--input your_data.jsonl` instead of `--dataset` for custom paired data.

### 3. Run full prefill, direct reuse, or Repeat-prefix

Run these commands in the `latent-remedy` environment:

```bash
python run.py --dataset all330 --method full --model 4b \
  --reasoning no --kv-mode all --batch-size 4

python run.py --dataset all330 --method reuse --model 4b \
  --reasoning no --kv-mode all --batch-size 4

python run.py --dataset all330 --method repeat-prefix --model 4b \
  --reasoning no --kv-mode all --batch-size 4
```

- `--kv-mode all`: both A/B full-prefill sides or both cross-reuse directions.
- `--reasoning yes`: explicit reasoning prompts; both modes keep Qwen's `enable_thinking=False`.
- `--limit 2 --dry-run`: inspect a small run's command without loading a model.
- `--output-root`: choose a results directory. Use a fresh directory for each configuration; `--overwrite` replaces an existing run.

### 4. Train LatentRemedy

Train from scratch using three pairs per training family:

```bash
python train_prefix_adapter_repeat.py \
  --input data/benchmark/generated240/train_180classes_30each.jsonl \
  --model 4b --model-path "$MODEL_PATH" --output results/adapter_train180 \
  --train-records-per-dataset 3 --train-reasoning no \
  --teacher-mode full --teacher-trace generated \
  --latent-tokens 16 --adapter-bottleneck 256 --epochs 1 --train-batch-size 4
```

This minimal example uses generated full-prefill teacher traces. For all240 training, replace the input with [data/benchmark/generated240/full_240classes_30each.jsonl](data/benchmark/generated240/full_240classes_30each.jsonl) and choose another output directory. Keep `--train-records-per-dataset 3` for three pairs per family; `--use-all-records` instead trains on the complete pool. Test60 is held out only when training on train180.

### 5. Evaluate LatentRemedy

```bash
python latent_remedy/eval_prefix_adapter_repeat.py \
  --checkpoint results/adapter_train180/latest_prefix_adapter.pt \
  --input data/benchmark/generated240/test_60classes_30each.jsonl \
  --model-path "$MODEL_PATH" --output results/adapter_test60 \
  --reasoning no --max-new-tokens 128 --batch-size 4

python latent_remedy/eval_prefix_adapter_repeat.py \
  --checkpoint results/adapter_train180/latest_prefix_adapter.pt \
  --input data/benchmark/benchmark_final240_real_ood_330_20261006.jsonl \
  --model-path "$MODEL_PATH" --output results/adapter_all330 \
  --reasoning no --max-new-tokens 128 --batch-size 4
```

For reasoning evaluation, use `--reasoning yes --max-new-tokens 4096` and a separate output directory. Use `--help` on the training/evaluation entry points for additional options.

### 6. Run third-party baselines

In this anonymous artifact, RelayCaching runs serially. Launch in the same `latent-remedy` environment:

```bash
python run.py --dataset all330 --method relaycaching --model 4b \
  --reasoning no --kv-mode all --batch-size 1 \
  --output-root results/relay_serial
```

Replace `relaycaching` with `cacheblend` or `epic` to run those baselines. KVCOMM uses `--method kvcomm --batch-size 1` and additionally requires `--baseline` pointing to the full-prefill sample outputs.

### 7. Check the installation

```bash
python scripts/verify_public_data.py
python -m unittest discover -s tests -v
```

## Acknowledgements

The third-party baseline code in this repository is adapted from [RelayCaching](https://github.com/YingshengGeng/RelayCaching). We thank its authors for open-sourcing their codebase.

## License

The project code is released under the [MIT License](LICENSE). Third-party code, datasets, and model weights retain their original terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
