# A Target-Provenance and Metric-Decomposition Audit for Ranked NL2Vis

This repository contains the code, frozen experimental designs, machine outputs,
and derived results for the paper *A Target-Provenance and Metric-Decomposition
Audit for Ranked NL2Vis*. The manuscript and submission documents are
intentionally excluded.

The study audits how target-derived benchmark annotations can enter ranked
NL2Vis candidate pools. It separates privileged diagnostics from
per-instance-target-answer-free generation, candidate coverage, and ordering.
The retained evidence includes a locked development-holdout comparison across
four local-model families and three prompt conditions, a same-lineage
nvBench-v1 cross-release check, repeated temperature-0 runs, locked
development-holdout RRF and two LLM rerankers, candidate-level execution,
study-normal-form and component analyses, and four rendered case studies.
The latest revision also includes a machine-validated stage-audit schema, an
independent VisEval provenance diagnostic across five model families, and
Qwen prompt-paraphrase and schema-order sensitivities on the same locked
150-case development sample. The newest block varies only candidate
presentation order across three fixed permutations for two LLM rerankers.

## Repository structure

- `experiment/`: analysis and inference code, frozen designs, validation
  summaries, and environment records.
- `experiment/reviewer_revision_round2_20260814/`: the locked development
  design, strict cross-release adapter, and round-2 analysis code.
- `experiment/reviewer_revision_round3_20260815/`: locked reranking design,
  public-workflow source audit, release-lineage audit, and analysis code.
- `experiment/reviewer_revision_round4_20260815/`: standardized-completion
  execution and
  complete-pool reranker-protocol audit code.
- `experiment/reviewer_revision_round5_20260815/`: core/component-matched
  breadth, cost, normal-form validation, forensic replay, and eligible-pool
  reranking code plus the inference-fixed protocol.
- `experiment/reviewer_revision_round6_20260815/`: executable audit schema,
  independent VisEval provenance analysis, and locked prompt/schema robustness
  code and protocols.
- `experiment/reviewer_revision_round7_20260816/`: locked candidate-position
  protocol and positional-robustness inference and analysis code.
- `results/reviewer_revision_round2_20260814/`: retained machine outputs and
  derived tables for the locked holdout, cross-release check, reranking extension,
  and temperature-0 repeatability experiment.
- `results/reviewer_revision_round3_20260815/`: locked reranking outputs,
  candidate/list-validity results, component metrics, and lineage-audit results.
- `results/reviewer_revision_round4_20260815/`: candidate-level renderer
  outcomes and retained-run checks of reranker input and output handling.
- `results/reviewer_revision_round5_20260815/`: all-TAF executable/intersected
  reranking outputs, leave-self-out results, normal-form perturbation checks,
  cost summaries, and matched-breadth tables.
- `results/reviewer_revision_round6_20260815/`: retained VisEval paired inputs,
  Qwen perturbation outputs, case-level analyses, and schema-validation records.
- `results/reviewer_revision_round7_20260816/`: two new presentation-order runs
  per reranker and their design-weighted stability summaries.
- `results/figures/`: final empirical plots and four rendered benchmark cases
  in PDF and PNG formats.
- `results/external_transfer/`: an earlier external-adapter campaign retained
  only as an excluded diagnostic; see its exclusion notice.

## Environment

The machine experiments use local [Ollama](https://ollama.com/) models. Use
Python 3.10 or newer. Package versions used for analysis are recorded in
`requirements.txt`; model names, prompt hashes, seeds, decoding settings, and
environment identifiers remain with the run outputs and manifests.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the registered Ollama models before rerunning machine inference. Run a
small engineering screen first to verify model availability and JSON-schema
compliance; performance outcomes are not an exclusion criterion in the locked
design.

## Recomputing analyses

The complete retained outputs allow metric recomputation without rerunning
model inference. The round-2 README lists the exact design and analysis entry
points. Primary scripts include `analyze_locked_holdout.py`,
`analyze_external_v3.py`, `analyze_stability.py`, and `analyze_reranker.py`.
Each script exposes its parameters through `--help` and writes machine-readable
CSV/JSON outputs.

## Data and scope

Included benchmark-derived records contain only fields needed for the reported
experiments. Historical runtime paths are normalized or confined to manifests;
no credentials are included. This archive contains no manuscript source,
submission file, or participant data. No participant study was conducted.

## Persistent archive

The current immutable research snapshot is available through the version DOI
reported in the paper. Earlier versions remain accessible from the Zenodo
concept record.

## License

Code is released under the MIT License. Benchmark-derived data remain subject
to their original licenses and terms.
