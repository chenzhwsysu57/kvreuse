# Compact-data release checks (2026-10-12)

- Main: 66 CPU tests passed after package renaming and input cleanup.
- Batch: 75 cases discovered in the main environment (71 passed, four legacy
  cache cases skipped). All nine Relay-specific cases then passed without skips
  in the separate Transformers 4.52.4 environment.
- All six JSONL files were compacted from 65,099,772 to 29,784,168 bytes.
- The migration compared prompt text, gold labels, record order and effective
  output protocols before and after cleanup: all were unchanged.
- The manifest-independent verifier checked minimal fields, all record counts,
  the 240/180/60 family partition, 30 pairs per family, exact train/test multiset
  union and real-world slice identity (110 ArgKP, 110 Deal, 110 Job pairs).
- The dataset validator accepted the compact 330-pair real-world input.
- Runtime IDs are content-derived, consistent across duplicate corpus slices,
  and not stored in the shipped data. Source metadata and the data manifest are
  removed; original ID-keyed results/caches must not be resumed.
- Training/evaluation help and syntax compilation passed. Both README Relay
  command paths were dry-run tested with compact inputs; serial and batching
  flags were correctly forwarded. Data, package and README match across branches.

No data hash manifest is needed or shipped. Corpus validation is structural and
semantic; it is not a cryptographic integrity certificate. No fresh dependency
installation or full-model GPU training, inference, accuracy or TTFT experiment
was performed. Static batching remains diagnostic, not official paper TTFT.
CUDA wheels are host-specific. First-party code is MIT-licensed; ArgKP terms need
confirmation, and Deal data retains its CC BY-NC restrictions.