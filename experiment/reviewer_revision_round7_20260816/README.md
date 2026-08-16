# Reviewer revision round 7

This round addresses candidate-order sensitivity in the eligible full-pool LLM reranking experiment. The baseline permutation seed 91 is retained from round 5. Two new presentation permutations (17 and 43) use the same 150 cases, candidate pools, prompt content, decoding seed 91, temperature 0, and exact-length output contract.

The analysis reports design-weighted outcome ranges, pairwise top-one identity,
top-five set Jaccard, unanimous top-one identity, prefix completeness, and
selected presentation position. The comparison isolates presentation order; it
is a post-review positional-robustness sensitivity, not a new untouched
confirmatory study.

## Completed result

All six model-permutation cells contain 150 cases when the two new runs are
combined with the registered permutation-91 baseline. The canonical pool and
RRF order are identical across permutations. Each model has one expected empty
pool per permutation; every nonempty response parses and returns exactly
`min(5, pool_size)` distinct IDs.

| Reranker | Hit@1 range | Hit@5 range | LLM − RRF Hit@1 range | Mean pairwise top-1 identity | Mean top-5 Jaccard | Unanimous top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Mistral-Small 24B | 7.52–11.49% | 26.44–27.81% | −4.73 to −0.76 points | 50.04% | .720 | 35.16% |
| Qwen 3 14B | 9.55–14.90% | 28.13–29.19% | −2.70 to +2.65 points | 53.69% | .717 | 40.30% |

The registered single-permutation deficits are therefore not a sufficient
description of ordering behavior. Mistral remains below RRF across all three
orders, whereas Qwen spans both sides of the RRF baseline. Both rerankers change
their selected top-one candidate in roughly half of paired presentation
comparisons.

Published derived files are under
`results/reviewer_revision_round7_20260816/analysis/positional/`:
`permutation_summary.csv`, `pairwise_stability.csv`, `model_summary.csv`,
case-level rows, and a completion manifest.
