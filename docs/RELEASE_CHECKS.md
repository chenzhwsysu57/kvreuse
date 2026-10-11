# Packaging checks (2026-10-12)

These are repository-packaging checks, not a new accuracy or performance experiment.

## main

- 55 CPU unittest cases passed in the existing main environment.
- All shipped first-party adapter modules imported without the original source
  checkout on the import path.
- Training and adapter-evaluation `--help` entry points loaded successfully.
- Python syntax compilation passed.
- All six data files matched their release SHA-256 hashes.
- The full 240-family corpus exactly matched the union of the 180-family train
  and 60-family held-out corpora, with 30 pairs per family and no family overlap.
- Combined real-world evaluation exactly matched the ArgKP/Deal and Job slices:
  110 ArgKP, 110 Deal, 110 Job pairs.

## batch

- Main-environment discovery ran 64 cases: 60 passed; four legacy-cache cases
  correctly skipped because Transformers 5.x lacks `DynamicCache.key_cache`.
- All nine Relay-specific CLI/cache cases then passed, without skips, in the
  existing separate Relay environment using Transformers 4.52.4 and CPU tensors.
- The real Relay runner's `--help` loaded in that separate environment.
- Data integrity checks and syntax compilation passed again.
- The two changed vendored cache/Qwen3 files matched the accbaseline source.
- Existing `gpt_chat.py` instrumentation and all data bytes are unchanged from main.
- Diff whitespace check passed. No credentials, private account paths, weights,
  logs, results, environments or nested Git repositories were added to Git.

## Not validated

- No fresh Conda/pip environment was installed; requirements syntax and actual
  imports were checked against existing environments. CUDA wheel selection remains
  host-specific.
- No full-model GPU training, inference, accuracy, throughput, TTFT, or serial/batch
  numerical equivalence experiment was run as part of packaging.
- This is a local Git repository only: no GitHub remote was supplied or pushed.
- First-party licensing and ArgKP redistribution terms require owner confirmation
  before public release. Deal retains its CC BY-NC restriction.