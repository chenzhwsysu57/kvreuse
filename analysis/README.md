# Attention analysis

`plot_deal_prefix_attention.py` selects Deal directions where dense full
prefill is correct but RoPE-corrected direct reuse is wrong.  It then measures
how each shared-block token distributes attention between the source prefix
(the context that produced the reused KV) and earlier tokens in that block.

Run it on a GPU node in the `kvreuse` environment:

```bash
/home/czw/miniconda3/envs/kvreuse/bin/python analysis/plot_deal_prefix_attention.py --model 1.7b
```

The default inputs are the no-reasoning `benchmark_250` full and reuse results.
Use `--reasoning yes` after both corresponding reasoning runs complete.  The
output directory contains one PNG and one NPZ per selected direction, plus a
JSONL selection file and aggregate `summary.json`.

`prefix_share_of_history` is `prefix_mass / (prefix_mass + preceding_block_mass)`.
It deliberately excludes self-attention, which otherwise dominates some early
block-token rows and obscures the prefix-vs-history comparison.

Use `--attention-context target` to produce the corresponding target dense
prompt diagnostic; the default `source` is the context actually used when the
reused block KV was created.

## ArgKP dense-prefix attention maps

`analyze_argkp_prefix_attention.py` performs one normal dense eager prefill
per ArgKP record and prefix side.  It writes four plots per case: a curve and a
layer-by-token heatmap for the total prefix-vs-block/self attention partition,
and a curve and heatmap for the strongest single prefix-token response.

```bash
/home/czw/miniconda3/envs/kvreuse/bin/python \
  analysis/analyze_argkp_prefix_attention.py --model 1.7b
```

The default input is the complete 110-record `data/processed/argkp.jsonl`; it
therefore produces 220 side-specific analyses.  For a smoke test, add
`--max-samples 1`.  Each `attention_scores.npz` contains
`prefix_dependency[layer, block_token]`, its complementary
`block_self_dependency`, and `peak_prefix_response`, the maximum attention
edge over every head and prefix-token key.

For the later partial-recompute diagnostic, the analyzer emits only contiguous
token windows in `selective_recompute_tokens.jsonl`: it separately takes the
top eight semantic-mass positions and top eight semantic-peak-response
positions, then chooses one length-eight window with maximal coverage for each
set.  The union of the two windows is merged into contiguous spans; it never
selects scattered individual tokens.  Change the budget with
`--selection-top-k` and `--recompute-span-length`.

## Correct-versus-wrong reuse comparison

`compare_deal_reuse_attention.py` uses every Deal reuse direction, grouping
them by whether RoPE direct reuse is correct or wrong.  It produces normalized
group heatmaps, a wrong-minus-correct difference heatmap, per-case metrics, and
a bootstrap confidence interval for the difference in mean prefix dependence.

```bash
/home/czw/miniconda3/envs/kvreuse/bin/python analysis/compare_deal_reuse_attention.py --model 1.7b
```
