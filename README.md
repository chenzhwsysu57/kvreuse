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

## Current scope

This first milestone covers only pinned dataset downloads, deterministic construction,
schema validation, and compact human-audit examples. Model inference, full-recompute
dataset validation, and KV-cache reuse evaluation are intentionally deferred to later
milestones so they can be reviewed and committed separately.
