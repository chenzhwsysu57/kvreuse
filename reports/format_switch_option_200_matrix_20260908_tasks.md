# Control-A format-switch 200 matrix

**Dataset:** `data/benchmark/benchmark_synthetic_format_switch_option_200/all.jsonl` — 200 paired examples / 400 cross-prefix directions. This is the clean direct-option Control-A benchmark, not the previous raw-payload stress set.

**Common protocol:** Qwen3 1.7B or 4B; batch size 16 for batched reuse; greedy decoding; all prompts require the final answer in exactly one `\boxed{...}`. No-reasoning uses the boxed assistant prefill; reasoning uses the explicit visible-reasoning instruction and final boxed-answer requirement. `reuse+post` prepends the fixed post-block text: “Use the instructions given earlier in this request when responding.”

| # | Task ID | Model | Mode | Method | Dependency |
|---:|---|---|---|---|---|
| 1 | `1.7b-no_reasoning-full` | 1.7B | no reasoning | Full | — |
| 2 | `1.7b-no_reasoning-reuse` | 1.7B | no reasoning | Direct Reuse | — |
| 3 | `1.7b-no_reasoning-ours_post` | 1.7B | no reasoning | Reuse + post | — |
| 4 | `1.7b-no_reasoning-kvcomm` | 1.7B | no reasoning | KVCOMM | 1 |
| 5 | `1.7b-no_reasoning-relaycaching` | 1.7B | no reasoning | RelayCaching | — |
| 6 | `1.7b-no_reasoning-epic` | 1.7B | no reasoning | EPIC | — |
| 7 | `1.7b-no_reasoning-cacheblend` | 1.7B | no reasoning | CacheBlend | — |
| 8 | `1.7b-reasoning-full` | 1.7B | visible reasoning | Full | — |
| 9 | `1.7b-reasoning-reuse` | 1.7B | visible reasoning | Direct Reuse | — |
| 10 | `1.7b-reasoning-ours_post` | 1.7B | visible reasoning | Reuse + post | — |
| 11 | `1.7b-reasoning-kvcomm` | 1.7B | visible reasoning | KVCOMM | 8 |
| 12 | `1.7b-reasoning-relaycaching` | 1.7B | visible reasoning | RelayCaching | — |
| 13 | `1.7b-reasoning-epic` | 1.7B | visible reasoning | EPIC | — |
| 14 | `1.7b-reasoning-cacheblend` | 1.7B | visible reasoning | CacheBlend | — |
| 15 | `4b-no_reasoning-full` | 4B | no reasoning | Full | — |
| 16 | `4b-no_reasoning-reuse` | 4B | no reasoning | Direct Reuse | — |
| 17 | `4b-no_reasoning-ours_post` | 4B | no reasoning | Reuse + post | — |
| 18 | `4b-no_reasoning-kvcomm` | 4B | no reasoning | KVCOMM | 15 |
| 19 | `4b-no_reasoning-relaycaching` | 4B | no reasoning | RelayCaching | — |
| 20 | `4b-no_reasoning-epic` | 4B | no reasoning | EPIC | — |
| 21 | `4b-no_reasoning-cacheblend` | 4B | no reasoning | CacheBlend | — |
| 22 | `4b-reasoning-full` | 4B | visible reasoning | Full | — |
| 23 | `4b-reasoning-reuse` | 4B | visible reasoning | Direct Reuse | — |
| 24 | `4b-reasoning-ours_post` | 4B | visible reasoning | Reuse + post | — |
| 25 | `4b-reasoning-kvcomm` | 4B | visible reasoning | KVCOMM | 22 |
| 26 | `4b-reasoning-relaycaching` | 4B | visible reasoning | RelayCaching | — |
| 27 | `4b-reasoning-epic` | 4B | visible reasoning | EPIC | — |
| 28 | `4b-reasoning-cacheblend` | 4B | visible reasoning | CacheBlend | — |

## Dispatcher policy

The dispatcher is `scripts/run_format_option_matrix.py`. Its `while` loop samples all four GPUs once a second and requires five consecutive low-use samples before dispatch: utilization below 50% and used memory at most 12 GiB. It records every launch and completion in `dispatch.json`, never relaunches a task, and waits for Full output before launching its corresponding KVCOMM task. Conservative admission limits keep the estimated total below 23 GiB.
