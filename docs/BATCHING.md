# Relay batching in the public `batch` branch

## Scope and provenance

No dataset contents, splits, task IDs, labels, or public dataset aliases change.
The standalone Relay runner now defaults to the shipped `all330` benchmark,
not the unavailable private `data/processed/all.jsonl`. The unified entry point
still supports registered datasets, custom `--input`, safe output paths,
input snapshots/hashes, explicit overwrite, and all existing main-branch methods.
The `repeat-prefix` method is unchanged. No TTFT suite/run-mode is added.

RelayCaching base revision: `6b0e9d37b74d89cfb0fbad276790150a0d9c0b49`.
The batching source revision is `18e2d061ba9118dd0a5905bee77e400d7b0e7aec`;
the two changed vendored files are:

- [relay_utils.py](../third_party/RelayCaching/RelayCaching/utils/relay_utils.py)
- [fixed_qwen3_hijack.py](../third_party/RelayCaching/RelayCaching/utils/fixed_qwen3_hijack.py)

The vendor tree is ordinary tracked source, not a submodule. The existing local
[gpt_chat.py](../third_party/RelayCaching/RelayCaching/llm/gpt_chat.py) generation
timing instrumentation is preserved, not replaced by the batching checkout.
See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for licenses/provenance.

## Execution paths and restrictions

- Default `--relay-execution-path official-serial` keeps the existing upstream
  placeholder/selective-generation path. It requires `--batch-size 1` and
  `--warmup-batches 0`; other values would otherwise be ignored.
- `static-batched` enables the static shared-block adapter for **RelayCaching,
  CacheBlend and EPIC only**. `--batch-size` counts source records; requested
  direction batch size is records times selected directions (two for `--kv-mode all`).
  It supports heterogeneous row lengths and per-row correction/selection.
- `--warmup-batches` runs untimed static batches and is not forwarded to other
  main-branch runners. Non-default Relay controls on non-Relay methods are rejected.
- `--epic-prefix-tokens` defaults to 16; changing it requires `--method epic`.
  `--relay-start-layer`, `--relay-det-layer`, `--relay-end-layer`,
  `--relay-tau-dev`, `--relay-tau-inf`, and `--relay-suffix-tokens` are forwarded
  to the Relay runner and recorded in the unified run config. Layer counts must
  also fit the chosen model; the CLI validates only model-independent ordering.
- Use positive `--max-new-tokens`. The unified defaults remain 128 for no-reasoning
  and 4096 for reasoning; these are generation budgets, not batch/TTFT settings.
  Both prompts use `enable_thinking=False`; reasoning only adds an explicit prompt.

The Qwen3 hijack uses **SDPA**. SDPA does not expose attention weights;
`output_attentions=True` is ignored by this hijack. Do not treat eager/FlashAttention
implementations as interchangeable numerical references. CacheBlend ranking and
adaptive selection can differ under padding/batch numerics. Static RelayCaching
is a diagnostic adapter, not a certification of all official influence/hidden-state
selection semantics or arbitrary `tau_inf` settings.

**`static-batched` timings are not official TTFT.** Warmups, source prefill,
correction, suffix processing and batched decoding form a different execution
schedule. Preserve their execution-path labels and compare accuracy separately;
do not relabel batch throughput or diagnostic phase times as official per-request TTFT.
Use separate output roots for execution paths or parameter variants: output naming
does not include batch size/path and existing nonempty outputs require explicit overwrite.

## Runnable examples

Run from the repository root. CPU-only command/config validation (no model imports):

```sh
python run.py --dataset all330 --method epic --model 4b --reasoning no \
  --kv-mode all --relay-execution-path static-batched --batch-size 2 \
  --warmup-batches 1 --epic-prefix-tokens 16 --limit 2 \
  --output-root results/batch-dry-run --dry-run
```

For actual inference, install [requirements-relay.txt](../requirements-relay.txt)
then [requirements.txt](../requirements.txt) in the same environment, provision the Qwen3 model
at the documented ModelScope cache location, and run on a CUDA GPU:

```sh
python run.py --dataset all330 --method epic --model 4b --reasoning no \
  --kv-mode all \
  --relay-execution-path static-batched --batch-size 2 --warmup-batches 1 \
  --max-new-tokens 128 --limit 2 --output-root results/batch-epic-smoke

python run.py --dataset all330 --method relaycaching --model 4b --reasoning no \
  --kv-mode a2b --limit 2 \
  --output-root results/official-relay-smoke
```

For CacheBlend or static RelayCaching, replace `--method epic` in the first GPU
example with `cacheblend` or `relaycaching` and choose a different output root.
No `--epic-prefix-tokens` override is needed for those methods.

CPU tests (no GPU/model loading):

```sh
python -m unittest discover -s tests -p 'test_relay_batch_cli.py' -v
python -m unittest discover -s tests -p 'test_relay_batch_cache.py' -v
```

The CLI tests use CPU mocks and the real runner parser AST. Cache tests use the
unchanged vendored implementation and real CPU tensors; they skip if torch or
transformers is absent or if `DynamicCache.key_cache` is unavailable (e.g.
transformers 5.x). Use the pinned Relay environment rather than modifying the
cache algorithm to accommodate incompatible dependencies.