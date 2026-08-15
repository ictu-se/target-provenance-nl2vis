# Target-Provenance Audits for Ambiguous NL2Vis

This repository contains the code, frozen experimental designs, machine outputs,
and derived results for the paper *Target-Provenance Audits for Ambiguous
NL2Vis: From Privileged Traces to Forward Alternatives*. The manuscript and
submission documents are intentionally excluded.

The study audits how target-derived benchmark annotations can enter ranked
NL2Vis candidate pools. It separates privileged diagnostics from
per-instance-target-answer-free generation, candidate coverage, and ordering.
The retained evidence includes a locked development-holdout comparison across
four local-model families and three prompt conditions, a strict nvBench-v1
eligible-subset check, repeated temperature-0 runs, pool-specific RRF and two
LLM rerankers, deterministic metric sensitivity analyses, and four rendered
case studies.

## Repository structure

- `experiment/`: analysis and inference code, frozen designs, validation
  summaries, and environment records.
- `experiment/reviewer_revision_round2_20260814/`: the locked development
  design, strict external adapter, and round-2 analysis code.
- `results/reviewer_revision_round2_20260814/`: retained machine outputs and
  derived tables for the locked holdout, external check, reranking extension,
  and temperature-0 repeatability experiment.
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

The immutable version-1.1.0 research snapshot is available at
[Zenodo](https://doi.org/10.5281/zenodo.21940669).

## License

Code is released under the MIT License. Benchmark-derived data remain subject
to their original licenses and terms.
