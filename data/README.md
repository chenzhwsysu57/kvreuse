# Shipped training and evaluation data

Only the current synthetic 240-family experiment corpus and the ArgKP, Deal or No
Deal, and Job Interview evaluation sets are included. There are no raw downloads,
model weights, inference outputs, historical datasets, HelpSteer, FANToM, HarmBench,
PKU-SafeRLHF, CaSiNo or Craigslist data.

## Synthetic experiment corpus

[benchmark/generated240](benchmark/generated240/) contains the complete 2026-10-08
replacement corpus: 240 families, 30 pairs per family; ten Deal/Job scene families
were replaced in the complete corpus. The 180/60 split is by family, not by pair.
The three files retain original bytes and are usable without a data generator or
private API credentials. Class names are the record `dataset` fields. The full
corpus is exactly the union of train and test, with no overlapping families.

The complete training pools are intentionally retained so both `all240` and
`train180` training can be reproduced. Choose the trainer's sampling options for
smaller pair-count experiments; do not silently use held-out families for train180.
All240 training includes every synthetic family; its separate real-world OOD
evaluation is the 330-record benchmark below.

## Real-world OOD evaluation

- [benchmark/benchmark_argkp_deal_220.jsonl](benchmark/benchmark_argkp_deal_220.jsonl):
  ArgKP 110 and Deal or No Deal 110 paired records.
- [benchmark/benchmark_job_interview_110.jsonl](benchmark/benchmark_job_interview_110.jsonl):
  Job Interview 110 paired records.
- [benchmark/benchmark_final240_real_ood_330_20261006.jsonl](benchmark/benchmark_final240_real_ood_330_20261006.jsonl):
  byte-preserved combined evaluation input, with all three sets (330 pairs / 660
  cross-prefix directions). `final240` denotes the experiment, not the record count.

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
- Synthetic records retain compiler, seed, family specification, oracle and audit
  metadata. First-party ownership/licensing is not inferred or changed by packaging.

## Integrity

[manifest.json](manifest.json) lists exact counts and SHA-256 hashes. Run
`python scripts/verify_public_data.py` to independently check every JSONL record,
the 240/180/60 family partition, 30 pairs per family, train/test membership, and
identity of the combined real-world input. This check uses only the Python standard
library and does not load any model.