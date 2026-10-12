# Compact-data release checks (2026-10-12)

- Main: 66 CPU tests passed after package renaming and input cleanup.
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
- Training/evaluation help, syntax compilation, README command construction and
  separate Relay CLI/cache tests are checked when syncing both branches.

No data hash manifest is needed or shipped. Corpus validation is structural and
semantic; it is not a cryptographic integrity certificate. No fresh dependency
installation or full-model GPU training, inference, accuracy or TTFT experiment
was performed. Static batching remains diagnostic, not official paper TTFT.
CUDA wheels are host-specific. Licensing still needs owner review: first-party
code has no chosen license, ArgKP terms need confirmation, and Deal is CC BY-NC.