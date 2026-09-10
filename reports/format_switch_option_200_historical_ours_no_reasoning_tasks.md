# Historical Ours comparison — Control-A, no reasoning

Dataset: `data/benchmark/benchmark_synthetic_format_switch_option_200/all.jsonl` (200 pairs / 400 directions). Each run uses Qwen3 no-thinking mode, greedy decoding, `max_new_tokens=128`, and requires one final `\boxed{...}` answer.

This is a correction to the prior matrix: its `reuse+post` used an ad-hoc synthetic suffix and is not included here.

| GPU | Model | Method | Exact post-block construction |
|---:|---|---|---|
| 0 | 1.7B | historical `ours_post` | `Current task objective (takes priority): <target_prefix>` followed by `Use the preceding document only according to this objective.` |
| 1 | 4B | historical `ours_post` | Same target-prefix bridge |
| 2 | 1.7B | historical `tail16_post_recompute` | Full target-prefix bridge plus cached-state precaution; target-recompute final 16 shared-block tokens |
| 3 | 4B | historical `tail16_post_recompute` | Same Tail-16 plus bridge |

The historical `ours_post` prompt is restored in `scripts/run_ours_bridge_reuse.py`; `tail16_post_recompute` uses `with_post_task_restatement()` in `scripts/run_direct_reuse.py`.
