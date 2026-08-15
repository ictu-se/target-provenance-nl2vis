# Canonical machine-experiment results

Integrity status: **complete**. Expert evidence is not included; the legacy synthetic ratings remain excluded.

## Legacy privileged diagnostic

On 751 test records, `step_6.answer` equals the canonical gold set in 92.81% and contains at least one gold in 97.20%. The privileged learned reranker reaches Hit@1=96.14% and Hit@5=97.47%; these are not end-to-end results.

## Cross-family forward generation (design-weighted 150-case sample)

| Model | Condition | n | Any valid % | Hit@1 % | Hit@5 % | Top-1 graded | Best-5 graded | Seconds/case |
|---|---|---|---|---|---|---|---|---|
| gemma3:27b | direct | 150 | 55.89 | 3.40 | 8.09 | 0.654 | 0.702 | 10.68 |
| llama3.2:3b | direct | 150 | 62.62 | 0.00 | 0.00 | 0.505 | 0.560 | 7.64 |
| mistral-small:24b | direct | 150 | 81.04 | 0.00 | 0.00 | 0.726 | 0.773 | 11.84 |
| qwen3:14b | direct | 150 | 54.25 | 0.00 | 1.34 | 0.710 | 0.737 | 3.49 |
| gemma3:27b | staged | 150 | 69.22 | 19.59 | 19.59 | 0.745 | 0.746 | 46.03 |
| mistral-small:24b | staged | 150 | 48.30 | 7.43 | 7.43 | 0.703 | 0.710 | 25.45 |
| qwen3:14b | staged | 150 | 49.83 | 11.58 | 12.93 | 0.732 | 0.750 | 14.53 |

## Qwen-14B full 751-case forward test

| Condition | n | Any valid % | Hit@1 % | Hit@5 % | MRR | Top-1 graded |
|---|---|---|---|---|---|---|
| direct | 751 | 60.32 | 0.40 | 1.20 | 0.007 | 0.721 |
| staged | 751 | 52.06 | 11.19 | 11.98 | 0.115 | 0.744 |

## Target-answer-free pooled reranking (design weighted)

| Reranker | n | Raw union | Valid pool | Pool oracle % | RRF H@1 % | RRF H@5 % | LLM H@1 % | LLM H@5 % | LLM graded |
|---|---|---|---|---|---|---|---|---|---|
| mistral-small:24b | 150 | 14.20 | 6.74 | 9.43 | 0.67 | 7.43 | 2.71 | 8.76 | 0.687 |
| qwen3:14b | 150 | 14.20 | 6.74 | 9.43 | 0.67 | 7.43 | 2.01 | 6.71 | 0.681 |

## Direct-IR disposition

Strict structured generation is unsupported: All 5,162 saved predictions fail strict JSON parsing. The legacy evaluator labels a row parseable when a regular expression finds a mark token, then extracts slots from malformed text. This does not support the manuscript claim of fully parseable IR. The direct-IR branch is excluded from the revised scope.

## Expert evidence

Deferred by instruction. No synthetic expert rating is used in these results.
