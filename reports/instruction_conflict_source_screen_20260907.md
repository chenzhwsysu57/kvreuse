# Instruction-conflict source intake screen

**Date:** 2026-09-07
**Target protocol:** Qwen3-1.7B, no reasoning, greedy boxed output; compare full A/B with A→B and B→A direct KV reuse.

## Batch-runtime gate

The imported batch-consistency probe did **not** pass. On its controlled reasoning case, serial and batch-size-2 generations differed, and several truncated or divergent generations did not yield the same parsed answer. Batch acceleration is therefore excluded from correctness screening until a no-reasoning, task-representative equivalence test passes. Screens below use the existing serial direct-reuse runner.

## Provenance snapshots

| Source | Pinned revision | License | Downloaded material |
|---|---|---|---|
| IHEval | `89d71d7b9522740e0d1c871752fd3cdbcee13131` | CC BY-NC-ND 4.0 | complete 63-file snapshot, 30.1 MB |
| Control Illusion | `9c08497e67caa123c89d3f1e049859164646a632` | repository does not declare a top-level license | source code plus 600 normal, 600 reversed, and 600 rich-context generated records |
| ConInstruct | `03251f209e16ee7e09867bff6030d1fd71156504` | CC BY 4.0 | 100 instructions, each carrying 7–9 conflict annotations |
| IH-Challenge | `056b7d94345dd4f8049da75bd70617d8928ac586` | Apache-2.0 | README plus 1,900-row `single-constraint` shard; larger shards intentionally deferred |

## IHEval: language detection versus summarization screen

The selected strong-conflict split contains 240 rows. The corrected source-grounded construction retains 160 rows where the official language label and reference summary are distinct. Its `prefix_a` contains only the language-detection task, `prefix_b` contains only the source summary instruction, and `shared_block` contains only the source news text. This is sufficient for a 110-record benchmark. It is still a **hierarchy-as-two-target-tasks proxy**, because the current runner does not preserve system/developer/user message roles.

The earlier test that placed both instructions into `shared_block` was invalid and is superseded by the following corrected 110-record screen:

| Measure | Result |
|---|---:|
| Full language target | 89.1% |
| Full summary target | 87.3% |
| Reuse A→B | 75.5% |
| Reuse B→A | 75.5% |
| Full target average | 88.2% |
| Reuse average | 75.5% |
| Average change | **-12.7 pp** |

**Decision: retain as a candidate.** Both targets are independently solvable and direct reuse lowers each target by 11.8–13.6 percentage points. It should be described as a role-conditioned language-detection versus summarization task, not as a faithful native-message hierarchy benchmark.

## Control Illusion

The source supplies 600 conflict records in each normal/reversed layout and 600 rich-context variants across six families: language, case, word-length, sentence-count, forbidden-keyword, and keyword-frequency constraints. Its evaluator checks whether an *open-ended generated response* meets each constraint; the data provides no paired reference responses or unique discrete gold answers.

**Decision: do not run the current full/reuse protocol.** Turning the two constraints themselves into options would make the prefix alone determine the answer and would not test shared-context reasoning. It needs a separate candidate-generation and verifier pipeline before it can be considered.

## ConInstruct

There are 100 base instructions and 864 annotated conflicts over nine conflict categories. Like Control Illusion, the source supplies conflicting constraints, not source-backed paired candidate responses or unique A/B answers. Multiple conflicts per base instruction are not independent shared-context examples.

**Decision: do not run the current full/reuse protocol.** It is too small at the instruction level and lacks the required labels. It can be revisited only if a verifier produces multiple independently valid candidate responses per constraint while preserving unique gold and sufficient diversity.

## IH-Challenge

The inspected 1,900-row shard covers 19 task types, 100 rows each. It includes attacker prompts, defender templates, and arbitrary Python grader code. Static AST inspection found only simple string/regex/JSON operations in this shard, but the tasks explicitly optimize attacks against privileged instructions. Dataset grader code was not executed.

**Decision: out of scope for a two-sided conflict benchmark.** A B-side task that pursues the attack objective would operationalize adversarial bypass behavior. Safe defender-only evaluation could be useful in a separate safety setting, but it does not supply two appropriate competing targets for this project.

## Next action

Retain the pinned raw snapshots. The corrected IHEval construction is ready for manual review and potential integration; Control Illusion, ConInstruct, and IH-Challenge are not candidates under the current task definition.
