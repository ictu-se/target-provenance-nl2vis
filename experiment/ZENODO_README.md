# Experimental archive: target-provenance audits for ambiguous NL2Vis

This machine-only archive supports the experiments reported in the article
“Target-Provenance Audits for Ambiguous NL2Vis: From Privileged Traces to
Forward Alternatives.” It contains generation and analysis code, registered
sample indices, model outputs, validation records, aggregate results, and
environment/version records.

The archive does not contain the manuscript, submission documents, or human
participant data. No participant study was conducted.

## Contents

- Python scripts in the archive root reproduce validation, exact and graded
  metrics, stratified estimates, stability analyses, pool construction, RRF,
  LLM reranking, and figures.
- `design/` records the fixed 150-case design and smoke decisions.
- `out/f/` contains leakage-free forward-generation outputs.
- `out/r/` and `out/r_major/` contain pool-specific reranking outputs.
- `out/a/`, `out/major_audit/`, and `out/final/` contain analysis-ready
  results and manuscript-facing aggregates.
- `out/env.json` and run manifests record exact software, model, prompt, and
  dataset versions.
- `ext_v1_20260810/` contains the derived external single-gold transfer set
  and its selection record.

The nvBench source datasets retain their original licenses and citations.
This deposit should be cited by its Zenodo DOI.
