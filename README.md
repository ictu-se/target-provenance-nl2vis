# Target-Provenance NL2Vis Evaluation

This repository contains the code, experimental designs, machine outputs, and derived results for the paper *Ambiguity-Aware Evaluation for Natural-Language-to-Visualization Systems*. The manuscript itself is intentionally excluded.

The study evaluates a stage-aware protocol that separates target recovery from downstream visualization construction. It includes paired direct-versus-staged runs, cross-family local-model comparisons, repeated-seed analyses, reranking ablations, four rendered case studies, and an external 105-case transfer study.

## Repository structure

- `experiment/`: experiment scripts, frozen designs, retained outputs, validation summaries, and analysis tables.
- `experiment/ext_v1_20260810/`: external-transfer design and source cases.
- `results/external_transfer/full_qwen/`: complete direct and staged Qwen outputs for the 105 external cases.
- `results/external_transfer/full_analysis/`: aggregate and paired external-transfer results.

## Environment

The machine experiments use local [Ollama](https://ollama.com/) models. Python package versions used for the analysis environment are recorded in `requirements.txt`; model names, prompts, seeds, and generation settings are retained with the run outputs and manifests.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the required Ollama models before rerunning machine inference. A short smoke run is recommended before a full campaign because model availability and runtime requirements vary by host.

## Reproducing analyses

Run commands from `experiment/`. The principal analysis entry points are:

```bash
python validate_outputs.py
python make_final_outputs.py
python make_major_revision_figures.py
python analyze_forward.py --help
python analyze_reranker.py --help
```

The external-transfer aggregate results are already retained under `results/external_transfer/full_analysis/`. The complete retained outputs allow metric recomputation without rerunning model inference.

## Data and scope

The included benchmark-derived records contain only the fields required for the reported experiments. Paths in historical runtime manifests have been normalized to portable placeholders. This archive contains no manuscript source, submission file, credentials, or personally identifying participant data.

## Persistent archive

A versioned research snapshot is also available at [Zenodo](https://doi.org/10.5281/zenodo.16752270).

## License

Code is released under the MIT License. Benchmark-derived data remain subject to their original licenses and terms.
