---
language:
- en
license: other
license_name: mixed-source-terms
license_link: https://huggingface.co/datasets/chenzhwsysu57/ConflictInstruct/blob/main/README.md#licensing
size_categories:
- 1K<n<10K
tags:
- kv-cache
- multi-agent
- conflicting-instructions
- latent-remedy
pretty_name: ConflictInstruct
configs:
- config_name: synthetic
  default: true
  data_files:
  - split: train
    path: benchmark/generated240/train_180classes_30each.jsonl
  - split: test
    path: benchmark/generated240/test_60classes_30each.jsonl
- config_name: all240
  data_files:
  - split: train
    path: benchmark/generated240/full_240classes_30each.jsonl
- config_name: real_ood
  data_files:
  - split: test
    path: benchmark/benchmark_final240_real_ood_330_20261006.jsonl
- config_name: argkp_deal
  data_files:
  - split: test
    path: benchmark/benchmark_argkp_deal_220.jsonl
- config_name: job
  data_files:
  - split: test
    path: benchmark/benchmark_job_interview_110.jsonl
---

# ConflictInstruct

Paired conflicting-instruction tasks for evaluating non-prefix KV cache reuse and
LatentRemedy. Each record shares the same context block between two instructions
with different correct answers.

**Code:** https://github.com/chenzhwsysu57/kvreuse

## Data configurations

| Configuration | Split | Pairs | Description |
|---|---|---:|---|
| `synthetic` | train / test | 5,400 / 1,800 | 180 training / 60 held-out synthetic families |
| `all240` | train | 7,200 | All 240 families; includes the synthetic test families |
| `real_ood` | test | 330 | ArgKP, Deal or No Deal, and Job Interview, 110 each |
| `argkp_deal` | test | 220 | ArgKP and Deal component input |
| `job` | test | 110 | Job Interview component input |

Every synthetic family has 30 pairs. The synthetic train and test splits are
family-disjoint. `all240` is their exact union, not an additional independent
dataset. `real_ood` is the union of `argkp_deal` and `job`; do not concatenate all
configurations and count duplicated records as independent examples. A pair has
two possible cross-reuse evaluation directions.

The synthetic corpus is the complete 2026-10-08 replacement version used by the
project, with ten Deal/Job-style scene families replaced. Synthetic family names
are compact anonymous identifiers (`family_001` through `family_240`).

## Load the data

```python
from datasets import load_dataset

synthetic = load_dataset("chenzhwsysu57/ConflictInstruct", "synthetic")
train = synthetic["train"]
test = synthetic["test"]

ood = load_dataset("chenzhwsysu57/ConflictInstruct", "real_ood", split="test")
argkp = ood.filter(lambda row: row["dataset"] == "argkp")
deal = ood.filter(lambda row: row["dataset"] == "deal_or_no_deal")
job = ood.filter(lambda row: row["dataset"] == "job_interview")
```

The `datasets` loader may add a nullable `output_protocol` column where the source
JSONL omits it. Treat a missing/null protocol as `option_letter`. The LatentRemedy
repository reads the original JSONL directly.

## Record format

| Field | Meaning |
|---|---|
| `dataset` | Synthetic family or real-world dataset label |
| `prefix_a`, `prefix_b` | The two conflicting task instructions |
| `shared_block` | Identical context shared by both tasks |
| `question` | Query following the context |
| `gold_a`, `gold_b` | Distinct correct answers under the two prefixes |
| `output_protocol` | Optional non-default answer parsing format |

The default answer format is `option_letter`. Other formats include `raw_payload`,
`final_payload`, `json_answer`, and `bracket_payload`. Preserve them during
evaluation. IDs, source identifiers, generation/audit metadata, and data manifests
are not stored. The code derives stable runtime IDs and uses exact-match scoring.

## Training and evaluation

For train180 experiments, sample the desired number of pairs per family only from
`synthetic/train`; keep `synthetic/test` out of training and checkpoint selection.
The project's three-pair setting selects 540 train180 pairs or 720 all240 pairs.
For all240 training, the synthetic test families are no longer held out: use the
separate real-world OOD input for cross-dataset evaluation.

Both no-reasoning and explicit-reasoning prompt modes keep Qwen native thinking
disabled (`enable_thinking=False`). See the code repository for actual training
and inference commands. Publishing these data does not establish GPU performance,
accuracy or TTFT equivalence under a different model/runtime.

## Sources

- **ArgKP:** IBM [KPA_2021_shared_task](https://github.com/IBM/KPA_2021_shared_task),
  based on the ArgKP work by Bar-Haim et al. The original documentation is retained
  at [notices/ARGKP_README.md](notices/ARGKP_README.md).
- **Deal or No Deal:** Facebook Research
  [end-to-end-negotiator](https://github.com/facebookresearch/end-to-end-negotiator).
  The original license is retained at [notices/DEAL_LICENSE](notices/DEAL_LICENSE).
- **Job Interview:**
  [negotiation-breakdown-detection](https://github.com/gucci-j/negotiation-breakdown-detection),
  revision `d4c2bf63b4da95b342fd952065f9ad3e97179134`.
  The original license is retained at [notices/JOB_LICENSE](notices/JOB_LICENSE).

## Licensing

This dataset repository has **mixed source terms**, not a blanket MIT license.
The software repository's MIT license does not override source dataset rights.

- **Deal data is CC BY-NC 4.0:** retain attribution and observe the noncommercial
  restriction, including in the combined `real_ood` configuration.
- **Job Interview data retains its upstream MIT license.**
- **ArgKP:** the retained upstream README does not specify a complete standalone
  dataset license. Check IBM's applicable dataset/shared-task terms before
  redistribution or commercial use. This release grants no additional rights to
  the source data.
- Synthetic records are project-generated; no upstream data license is inferred
  from the project's software license. Contact the maintainers for clarification
  of dataset-specific rights where needed.

## Citation

```bibtex
@misc{chen2026latentremedy,
  title={LatentRemedy: Efficient Remedy For Non-Prefix KV Cache Reuse in Multi-Agent Systems},
  author={Zhiwei Chen and Zhongtao Rao and Waikit Xiu and Yuxiang Zhang and Shuting Cui and Jian Chen and Tianxiang Zhao},
  year={2026},
  eprint={2610.18014},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2610.18014}
}
```

Also cite and comply with the original real-world datasets when using them.