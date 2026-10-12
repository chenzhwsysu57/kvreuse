# Shipped training and evaluation data

Only the current synthetic 240-family experiment corpus and the ArgKP, Deal or No
Deal, and Job Interview evaluation sets are included. There are no raw downloads,
model weights, inference outputs, historical datasets, HelpSteer, FANToM, HarmBench,
PKU-SafeRLHF, CaSiNo or Craigslist data.

## Synthetic experiment corpus

[benchmark/generated240](benchmark/generated240/) contains the complete 2026-10-08
replacement corpus: 240 families, 30 pairs per family; ten Deal/Job scene families
were replaced in the complete corpus. The 180/60 split is by family, not by pair.
The files are compacted JSONL, not original-byte copies, and are usable without a
data generator or private API credentials. Synthetic `dataset` values are anonymous
`family_001` through `family_240`, consistently mapped across full/train/test.
Prompts, gold answers, record groupings and family-level split membership are
unchanged. The full corpus is the train/test union, with no overlapping families;
post-cleanup verification is pending with the main agent.

The complete training pools are intentionally retained so both `all240` and
`train180` training can be reproduced. Choose the trainer's sampling options for
smaller pair-count experiments; do not silently use held-out families for train180.
Use `--train-records-per-dataset 3` for three pairs per family, not
`--use-all-records`, which would select all 30. The minimal training example in
[../README.md](../README.md) is not an exact paper-configuration claim.
All240 training includes every synthetic family; its separate real-world OOD
evaluation is the 330-record benchmark below.

## Real-world OOD evaluation

- [benchmark/benchmark_argkp_deal_220.jsonl](benchmark/benchmark_argkp_deal_220.jsonl):
  ArgKP 110 and Deal or No Deal 110 paired records.
- [benchmark/benchmark_job_interview_110.jsonl](benchmark/benchmark_job_interview_110.jsonl):
  Job Interview 110 paired records.
- [benchmark/benchmark_final240_real_ood_330_20261006.jsonl](benchmark/benchmark_final240_real_ood_330_20261006.jsonl):
  compact combined evaluation input, with all three sets (330 pairs / 660
  cross-prefix directions). `final240` denotes the experiment, not the record count.

Real-world `dataset` values remain the short names `argkp`, `deal_or_no_deal`, and
`job_interview`; runner aliases `deal` and `job` select the latter two values.

## Compact record schema and runtime compatibility

The seven core fields are `dataset`, `prefix_a`, `prefix_b`, `shared_block`,
`question`, `gold_a`, and `gold_b`. Retain optional `output_protocol` only when it
is not `option_letter`; omission means the default option-letter protocol. Keep
non-default protocols such as `json_answer` unchanged.

No `task_id`, `metric`, metadata, compiler, seed, family specification, oracle,
audit trace or other extra field is stored. At load time, the runtime derives a
path-independent `task_id` from canonical core content and an explicitly supplied
protocol, and defaults `metric` to `exact_match`.

The cleanup preserves prompt/gold content and group/split membership, not original
JSONL serialization, old hashes or old IDs. New runtime IDs cannot resume legacy
outputs or reuse old ID-keyed teacher/state caches. Use fresh results directories
and regenerate caches and direction-ID lists rather than mixing old and new runs.

## Provenance and terms

- **ArgKP**: IBM https://github.com/IBM/KPA_2021_shared_task;
  https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml.
  Original dataset/shared-task documentation is retained in
  [notices/ARGKP_README.md](notices/ARGKP_README.md). Its documentation does not state
  a complete standalone dataset license. Check IBM's dataset/shared-task terms
  before redistribution or commercial use; this release does not grant additional
  rights. Cite the original Bar-Haim et al. work referenced in the upstream README.
- **Deal or No Deal**: Facebook Research
  https://github.com/facebookresearch/end-to-end-negotiator. The actual data notice is
  **CC BY-NC 4.0**, retained in [notices/DEAL_LICENSE](notices/DEAL_LICENSE). Preserve
  attribution and observe the noncommercial restriction.
- **Job Interview**: https://github.com/gucci-j/negotiation-breakdown-detection,
  source revision `d4c2bf63b4da95b342fd952065f9ad3e97179134`, MIT license retained
  in [notices/JOB_LICENSE](notices/JOB_LICENSE). Its original application checkout
  is not needed for evaluation and therefore is not vendored.
- Synthetic compiler, seed, family specification, oracle and audit metadata are
  removed from shipped records. This cleanup does not establish or change
  first-party ownership/licensing.

## Integrity

The former data manifest has been deleted.
[../scripts/verify_public_data.py](../scripts/verify_public_data.py) does not depend
on a manifest and rejects data manifests if present. Run
`python scripts/verify_public_data.py` from the repository root to check the six
expected JSONL files, counts, minimal-field schema, 240/180/60 family partition,
30 pairs per family, anonymous family names, train/test isolation and multiset
membership, and the combined real-world input against its component files.
This uses only the Python standard library and does not load a model.

It is not a historical SHA-256 comparison, nor does it alone prove unchanged
prompt/gold content relative to pre-cleanup sources. The main agent must still
execute post-cleanup verification and compare preserved content/group/split
membership with the source data. Release checksum publication is a commitment,
not a completed check; see [../docs/RELEASE_CHECKS.md](../docs/RELEASE_CHECKS.md).