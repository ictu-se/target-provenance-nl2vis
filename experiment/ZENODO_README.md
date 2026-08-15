# Experimental archive: stage-resolved target provenance and exact-match evaluation in ranked NL2Vis

This machine-only archive supports the experiments reported in “A Stage-Resolved
Audit of Target Provenance and Exact-Match Evaluation in Ranked NL2Vis.” It contains generation
and analysis code, frozen sample indices, model outputs, validation records,
aggregate results, rendered experimental figures, and environment/version
records.

The archive does not contain the manuscript, submission documents, credentials,
or human-participant data. No participant study was conducted.

## Evidence groups

- Privileged diagnostics trace benchmark step answers into candidate pools and
  quantify stage-wise target containment and counterfactual removals.
- Historical forward, temperature-0.2, and pooled-reranking outputs are retained
  as exploratory because engineering reused the nvBench 2.0 test split.
- `reviewer_revision_round2_20260814/` contains the locked development screen,
  disjoint holdout, three-condition cross-family experiment, strict nvBench-v1
  eligible-subset check and temperature-0 repeatability evidence.
- `reviewer_revision_round3_20260815/` adds candidate-level conformity,
  component metrics, the public-workflow provenance check, cross-release
  overlap audit, and locked Qwen/Mistral reranking on three pools.
- `reviewer_revision_round4_20260815/` separates JSON parseability,
  Vega-Lite renderer execution, and study benchmark conformity, and
  verifies complete-pool exposure and prefix-only output handling for the
  retained reranker runs.
- `reviewer_revision_round5_20260815/` surfaces core and component-matched
  breadth, primary latency and token cost, validates the normal-form checker
  using 15,750 perturbations, replays the retained forensic record, and runs
  full and leave-self-out reranking on executable/intersected all-TAF pools.
- `reviewer_revision_round6_20260815/` adds an executable stage-audit schema,
  a paired independent-VisEval provenance diagnostic across five model
  families, and Qwen prompt-paraphrase/schema-order robustness outputs on the
  locked 150-case development sample.
- The earlier cross-release adapter campaign is retained with an explicit
  exclusion notice and is not part of the reported result.

## Analysis scope

The archive supports validation, exact specification reproduction,
core-equivalence sensitivity, graded fidelity, GoldRecall, exact-gain taxonomy,
stratified paired intervals, pool construction, RRF sensitivity, constrained
LLM reranking, cross-seed overlap, renderer execution, and reranker-protocol
checks. Benchmark source datasets retain their original licenses and citations.

Use the version DOI shown by the current Zenodo record when citing this archive.
