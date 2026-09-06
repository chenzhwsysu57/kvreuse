# Craigslist Bargains dialogue candidate and shared-context ablation

**Date:** 2026-09-06
**Model:** Qwen3-1.7B and Qwen3-4B
**Mode:** no reasoning; greedy boxed option output
**Method:** direct full prefill versus cross-prefix KV reuse

## Construction

`craigslist_dialogue` is a source-grounded buyer-versus-seller retrieval task. Each record asks for either the buyer's or seller's final explicit price proposal in the same accepted negotiation. The two candidates are those two proposals, so the gold option necessarily flips with the target role.

The builder accepts a record only when:

- the source declares the canonical role ordering `[buyer, seller]`;
- `agent_turn` identifies buyer as `0` and seller as `1`;
- all turn, utterance, intent, and price arrays align;
- the negotiation contains a source `accept` act;
- each role has a source-annotated final `init-price`, `counter-price`, or `offer` utterance;
- each retained proposal has exactly one numeric amount in its text, matching the source annotation; and
- buyer and seller final explicit proposals are different.

This yields **1,832** records. An independent post-build audit recomputed every rendered-dialogue SHA-256 and confirmed the recorded buyer/seller event indices against the raw source. Candidate option order is near balanced: `A→B` in 941 records and `B→A` in 891.

Two matched 60-record benchmarks use identical task IDs, candidate order, and labels:

- **Compact:** only the two final proposal utterances are shared; mean shared context 220 characters.
- **Full dialogue:** every nonempty dialogue turn is shared; mean shared context 699.5 characters (maximum 2,501).

## Results

| Context | Model | Full buyer | Full seller | Full target avg. | A→B reuse | B→A reuse | Reuse avg. | Avg. drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Compact final proposals | 1.7B | 51.7% | 70.0% | 60.8% | 58.3% | 41.7% | 50.0% | 10.8 pp |
| Full dialogue | 1.7B | 41.7% | 68.3% | 55.0% | 63.3% | 33.3% | 48.3% | 6.7 pp |
| Compact final proposals | 4B | 68.3% | 91.7% | **80.0%** | 48.3% | 13.3% | **30.8%** | **49.2 pp** |
| Full dialogue | 4B | 45.0% | 73.3% | 59.2% | 68.3% | 31.7% | 50.0% | 9.2 pp |

All runs parsed an answer for every output and none reached the generation limit.

## Decision

The **compact final-proposal context** is a viable Craigslist candidate on Qwen3-4B: it has a large source-grounded corpus, strong full accuracy (80.0% aggregate), and a 49.2-point cross-prefix reuse decline. A 110-record benchmark was prepared for follow-up.

The **complete dialogue must not be used as its shared block**. It adds roughly 480 characters on average, reduces full accuracy by 20.8 points on the 4B matched slice, and reduces the observed reuse drop from 49.2 to 9.2 points. It is therefore harmful noise for this task rather than useful causal evidence.

The task is a role-conditioned factual retrieval conflict, not a negotiation-utility or real-final-sale-price task. This limitation should be stated if it is included in a benchmark.
