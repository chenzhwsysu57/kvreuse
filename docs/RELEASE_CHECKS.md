# Compact-data release checks (2026-10-12)

- Main: 66 CPU tests passed after package renaming and input cleanup.
- Batch: 75 cases discovered in the main environment (71 passed, four legacy cache cases skipped). All nine Relay-specific cases then passed without skips in the separate Transformers 4.52.4 environment. This is the historical split-environment check, not a rerun under the unified setup below.
- All six JSONL files were compacted from 65,099,772 to 29,784,168 bytes.
- The migration compared prompt text, gold labels, record order and effective output protocols before and after cleanup: all were unchanged.
- The manifest-independent verifier checked minimal fields, all record counts, the 240/180/60 family partition, 30 pairs per family, exact train/test multiset union and real-world slice identity (110 ArgKP, 110 Deal, 110 Job pairs).
- The dataset validator accepted the compact 330-pair real-world input.
- Runtime IDs are content-derived, consistent across duplicate corpus slices, and not stored in the shipped data. Source metadata and the data manifest are removed; original ID-keyed results/caches must not be resumed.
- Training/evaluation help and syntax compilation passed. Both README Relay command paths were dry-run tested with compact inputs; serial and batching flags were correctly forwarded. Data, package and README matched across branches at this historical check.

No data hash manifest is needed or shipped. Corpus validation is structural and semantic; it is not a cryptographic integrity certificate. No fresh dependency installation or full-model GPU training, inference, accuracy or TTFT experiment was performed. Static batching remains diagnostic, not official paper TTFT. CUDA wheels are host-specific. First-party code is MIT-licensed; ArgKP terms need confirmation, and Deal data retains its CC BY-NC restrictions.

## Unified-environment check (2026-10-13)

The supported setup uses one conda environment named `latent-remedy` with Python 3.10 for training, evaluation, and RelayCaching/CacheBlend/EPIC. Install the Relay base first with `python -m pip install -r requirements-relay.txt`, then our own additions with `python -m pip install -r requirements.txt`; the main requirements include `-r requirements-relay.txt`. No second environment or `RELAY_PYTHON` variable is needed. [run.py](../run.py) defaults to `sys.executable`; `--relay-python` remains an optional interpreter override.

- Tested environment: the existing Relay environment with Python 3.10.21, PyTorch 2.7.1+cu128, Transformers 4.52.4, Accelerate 1.5.2, NumPy 1.26.4, and ModelScope 1.40.1.
- All 66 `main` CPU tests passed in this environment.
- All 75 `batch` CPU tests passed in this same environment, including the legacy-cache tests, with no skips.
- Training, evaluation, and Relay entry-point `--help` checks passed in this environment.
- Pip's requirements dry-run reported all unified dependencies satisfied; no package installation or environment modification was performed.

Limits: this check reused an existing environment; it did not perform a fresh install using the documented dependency order. No GPU training, inference, accuracy, TTFT, or throughput run was performed. Static batching remains diagnostic, not official paper TTFT.