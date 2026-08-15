# Experimental methods and retained evidence

This directory contains the auditable machine experiments for the
target-provenance study. Privileged diagnostics, historical post-selection
experiments, and locked primary experiments remain distinguishable in file
names, designs, and analysis manifests.

## Evidence policy

- Query and schema are the only per-instance inputs permitted in forward
  generation.
- Benchmark step answers are privileged annotations and are used only for the
  provenance diagnostic or oracle analyses.
- Forward conditions are described as per-instance-target-answer-free, not as
  leakage-free in an unrestricted sense; their prompts were designed with
  knowledge of the benchmark grammar.
- Synthetic preview ratings and malformed Direct-IR outputs are excluded from
  scientific results.
- Run outputs retain the model, condition, seed, prompt hash, decoding settings,
  timing, parsing status, candidates, and validation outcomes.

## Data

Scripts locate the workspace-relative nvBench 2.0 checkout by default. In a
standalone copy, pass `--data-dir` to a directory containing the released split
files. The exact dataset commit, split counts, and checksums are recorded in the
round-2 provenance manifest. The strict nvBench-v1 derived subset and its
selection record are included because they are required to reproduce the
external analysis.

The audited secondary source implementation is not redistributed. Forward
experiments do not depend on it. The repository instead retains the audit code,
stage-wise aggregates, counterfactual results, and worked-example evidence.

## Main scripts

- `common.py`: parsing, canonicalization, validation, and exact/graded metrics.
- `audit_legacy.py`: six-step candidate ancestry and privileged containment.
- `run_forward_ollama.py`: resumable direct-basic, direct-rich, and staged-rich
  local-model runs.
- `analyze_forward.py`: exact/graded metrics and grouped summaries.
- `run_forward_reranker.py`: fixed-pool RRF and constrained LLM reranking.
- `analyze_reranker.py`: pool oracles, RRF/LLM results, and paired contrasts.
- `analyze_stability.py`: cross-seed top-one, set, valid-set, and rank overlap.
- `capture_environment.py`: software, hardware, model, data, prompt, and script
  provenance manifests.
- `reviewer_revision_round2_20260814/`: locked-design, strict-adapter,
  core-equivalence, exact-gain taxonomy, and sensitivity scripts.
- `reviewer_revision_round3_20260815/`: public-workflow and cross-release
  audits, locked reranking analysis, and empirical figure generation.
- `reviewer_revision_round4_20260815/`: renderer-execution and retained
  complete-pool reranker-protocol audits.

## Locked experimental design

- The primary prompt comparison uses a 30-case development engineering screen
  followed by a disjoint stratified 150-case development holdout.
- Gemma, Llama, Mistral, and Qwen run direct-basic, instruction-matched
  direct-rich, and staged-rich at temperature 0 on the same holdout.
- Exact or graded outcomes cannot remove a model or condition from the screen.
- The strict nvBench-v1 check samples 15 cases from each of seven chart families
  after deterministic eligibility filtering and adapter validation.
- Main-setting repeatability uses Qwen seeds 55, 101, and 202 at temperature 0
  for all three prompt conditions on the locked holdout.
- The historical nvBench 2.0 test campaign, stochastic temperature-0.2 runs,
  case selection, and pooled reranking remain exploratory because engineering
  previously reused that test split.
