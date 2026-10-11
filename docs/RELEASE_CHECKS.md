# Packaging checks (2026-10-12)

- `main`: 55 CPU unittest cases passed; training/evaluation help entry points and
  every shipped first-party adapter module imported successfully.
- `batch`: 64 cases discovered in the main environment (60 passed, four legacy
  cache cases skipped for Transformers 5.x). All nine Relay-specific CLI/cache
  cases then passed without skips in the separate Transformers 4.52.4 environment.
- The real serial and batch Relay runner help entry points loaded in that environment.
- Syntax compilation and diff whitespace checks passed.
- All six data files matched their release SHA-256 hashes. Full/train/test have
  240/180/60 families, 30 pairs each; full is exactly the train/test union with no
  family overlap. Real-world data has 110 pairs each for ArgKP, Deal and Job.
- Data bytes and existing Relay generation timing instrumentation are identical
  between branches. The two batching cache/Qwen3 files match the baseline source.
- No credentials, private account paths, weights, results, logs, environments,
  historical Git commits or nested Git repositories were included in Git.

No fresh dependency installation or full-model GPU experiment was run during
packaging. These checks do not establish accuracy, throughput, TTFT or serial/batch
numerical equivalence. CUDA wheels remain host-specific. No GitHub remote was
provided or pushed. First-party licensing and ArgKP redistribution terms require
owner confirmation; Deal data retains its CC BY-NC restriction.