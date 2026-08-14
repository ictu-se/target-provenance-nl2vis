# Cross-family smoke decisions

Corrected smoke campaign: seed 68, temperature 0, query plus SQLite-derived
schema only. Qwen used 14 cases (two per released family). Mistral and Llama
used seven cases (one per family). Gemma direct used 14; staged used the matched
seven-family subset. These are screening results, not manuscript effect estimates.

| Model | Condition | n | Hit@1 | Hit@5 | Top-1 macro | Best-5 macro | Any valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3 14B | direct | 14 | 0.000 | 0.000 | 0.836 | 0.858 | 0.286 |
| Qwen 3 14B | staged | 14 | 0.071 | 0.071 | 0.824 | 0.838 | 0.429 |
| Gemma 3 27B | direct | 14 | 0.000 | 0.071 | 0.784 | 0.839 | 0.500 |
| Gemma 3 27B | staged | 7 | 0.143 | 0.143 | 0.772 | 0.772 | 0.429 |
| Mistral-Small 24B | direct | 7 | 0.000 | 0.000 | 0.875 | 0.875 | 0.429 |
| Mistral-Small 24B | staged | 7 | 0.000 | 0.000 | 0.810 | 0.810 | 0.286 |
| Llama 3.2 3B | direct | 7 | 0.000 | 0.000 | 0.618 | 0.647 | 0.429 |
| Llama 3.2 3B | staged | 7 | 0.000 | 0.000 | 0.569 | 0.569 | 0.000 |

Decision: retain Qwen direct/staged for the full 105-case paired external run.
Do not expand Llama staged because it structurally degenerates (zero cases with
a valid candidate). Do not expand Mistral or Gemma: their smoke results add
cross-family format/fidelity evidence, while full runs would be costly and
scientifically redundant with the manuscript's existing 150-case cross-family
nvBench 2.0 experiment. Exact and graded results disagree enough to require both
metrics in the external report.

The invalid earlier adapter campaign is separately retained and excluded at
`_runs/p01_extv1/aborted_sort_omission/`.
