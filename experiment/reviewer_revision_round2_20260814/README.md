# Reviewer revision round 2 (2026-08-14)

This directory contains the machine evidence added in response to the major
review concerning prompt confounding, test-set reuse, exact-match semantics,
external-adapter validity, reranking scope, stability, and dataset provenance.

## Locked primary design

- `design/locked_protocol.json`: model, prompt, decoding, contrasts, and
  main-setting robustness extension locked before holdout inference.
- `design/dev_engineering_screen30.json`: development-only engineering screen;
  outcome scores cannot remove a run.
- `design/dev_locked_holdout150.json`: disjoint stratified holdout sampled from
  the remaining 720 development cases.
- `design/dataset_provenance.json`: release commit, split counts, checksums, and
  semantic equality to the committed JSON.
- `design/runtime_environment.json`: public hardware/software/model manifest.

The primary comparisons are direct-rich minus direct-basic and staged-rich
minus direct-rich. Staged-rich minus direct-basic is secondary because it
changes both instruction richness and decomposition.

## External transfer

`external_v3/` is the only admissible nvBench-v1 transfer set. Its adapter
retains supported binning and sorting, excludes unsupported filters,
distinct/min/max aggregation, limits, and ordering constructs, and has
a deterministic 21-case manual conversion audit. Earlier external campaigns
are explicitly marked excluded and must not appear in manuscript results.
`prepare_external_strict.py` reconstructs the eligible set and frozen balanced
sample; `analyze_external_v3.py` recomputes its reported results.

## Analyses

- `analyze_locked_holdout.py`: design-weighted full exact, core-equivalence,
  graded, validity, latency, paired contrasts, and exact-gain taxonomy.
- `analyze_external_v3.py`: strict external condition and paired analysis.
- `metric_equivalence.py`: shared core-equivalence and gain-taxonomy rules.
- `analyze_pool_sensitivity.py`: raw/valid pool distributions, duplicate-source
  provenance, and RRF constant sensitivity.
- `collect_dataset_provenance.py`: exact dataset snapshot manifest.
- `../analyze_stability.py`: main-setting and sampling-robustness summaries.

Historical nvBench-2.0 test-set forward and reranking results are retained as
exploratory diagnostics because model/prompt engineering previously reused that
test set. They are not the primary prompt-effect evidence.
