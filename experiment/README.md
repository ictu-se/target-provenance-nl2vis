# Reviewer-revision experiments

This directory contains the auditable replacement experiments for the rejected
Information Visualization submission.  The original submission snapshot is
left unchanged.

## Evidence policy

- `query + schema` are the only inputs allowed in deployable forward conditions.
- Benchmark `steps` are privileged annotations.  Results using them are reported
  only as diagnostic or oracle conditions.
- `step_6.answer` is never treated as a model prediction.
- The synthetic expert-rating CSV from the legacy project is excluded from all
  scientific analyses.
- Every output records input hashes, model name, seed, prompt hash, and timestamp.

## Data

Scripts locate the workspace-relative nvBench 2.0 checkout by default. In a
standalone copy, set `NVBENCH2_DATA_DIR` to the directory containing
`train.json`, `dev.json`, and `test.json`, or pass `--data-dir` where offered.
Set `NVBENCH2_TEST_FILE` for the operation-group post-hoc analysis.

The audited legacy source is not redistributed here. Set
`INTENTLENS_LEGACY_DIR` or pass `--legacy-dir` to the directory containing
`paper4_candidates.py` and its saved artifacts. Forward experiments do not
depend on the legacy source.

## Scripts

- `common.py`: parsing, canonicalization, exact and graded metrics.
- `audit_legacy.py`: six-step provenance, leakage, candidate-pool stages, and
  reproduction of the legacy heuristic/reranker.
- `run_forward_ollama.py`: resumable direct and forward-six-step Ollama runs.
- `analyze_forward.py`: exact/graded metrics, coverage curves, uncertainty, and
  grouped analyses.
- `make_stratified_sample.py`: fixed, proportional mark-family test sample with
  every non-empty stratum represented.
- `run_campaign.py`: resumable cross-family, full-test, and repeated-run campaign.
- `run_forward_reranker.py`: leakage-free complete-pool RRF and LLM reranking.
- `analyze_stratified_sample.py`: design-weighted estimates and stratified
  bootstrap intervals.
- `analyze_paired_conditions.py`: paired direct-versus-staged effects.
- `analyze_stability.py`: cross-seed top-one, set, and rank stability.
- `analyze_legacy_paired.py`: paired bootstrap and exact McNemar audit of the
  privileged legacy condition.
- `make_worked_example.py`: complete trace-to-pool-to-ranking diagnostic example.
- `make_audit_figures.py`: fixed-pool coverage and pool-stage figures.
- `capture_environment.py`: software, hardware, model-digest, data-hash, and
  script-hash manifest.

Generated files are written under `outputs/` and the short-path artifact tree
`out/` (needed because the manuscript directory is close to the Windows legacy
path-length limit). `outputs/campaign_status.json` is the canonical live status.

## Locked experimental design

- Legacy results use all 751 test records and are labeled privileged diagnostics.
- Cross-family forward comparison uses the fixed 150-record design in
  `design/forward_sample150.json`, sampled within canonical gold mark-family
  strata and analyzed with population weights.
- Qwen-14B direct and forward-staged conditions also run on all 751 test records.
- Qwen-14B stability uses three temperature-0.2 seeds on the same 150 records.
- DeepSeek-1.5B, Phi-3.5, and staged Llama-3.2 are smoke-only exclusions under
  the predeclared evidence in `design/smoke_decisions.json`.
