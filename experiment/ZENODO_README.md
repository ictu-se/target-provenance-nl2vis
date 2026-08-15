# Experimental archive: target-provenance audits for ambiguous NL2Vis

This machine-only archive supports the experiments reported in
“Target-Provenance Audits for Ambiguous NL2Vis: From Privileged Traces to
Forward Alternatives.” It contains generation and analysis code, frozen sample
indices, model outputs, validation records, aggregate results, rendered
experimental figures, and environment/version records.

The archive does not contain the manuscript, submission documents, credentials,
or human-participant data. No participant study was conducted.

## Evidence groups

- Privileged diagnostics trace benchmark step answers into candidate pools and
  quantify stage-wise target containment and counterfactual removals.
- Historical forward, temperature-0.2, and pooled-reranking outputs are retained
  as exploratory because engineering reused the nvBench 2.0 test split.
- `reviewer_revision_round2_20260814/` contains the locked development screen,
  disjoint holdout, three-condition cross-family experiment, strict nvBench-v1
  eligible-subset check, Mistral all-pool reranking extension, and
  temperature-0 repeatability evidence.
- The earlier external-adapter campaign is retained with an explicit exclusion
  notice and is not part of the reported external result.

## Analysis scope

The archive supports validation, exact specification reproduction,
core-equivalence sensitivity, graded fidelity, GoldRecall, exact-gain taxonomy,
stratified paired intervals, pool construction, RRF sensitivity, constrained
LLM reranking, and cross-seed overlap. Benchmark source datasets retain their
original licenses and citations.

Please cite version 1.1.1 as https://doi.org/10.5281/zenodo.21941099.
