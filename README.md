# Target-Provenance NL2Vis Evaluation

This repository contains the code, experimental designs, machine outputs, and derived results for the paper *Ambiguity-Aware Evaluation for Natural-Language-to-Visualization Systems*. The manuscript itself is intentionally excluded.

The study evaluates a stage-aware protocol that separates target recovery from downstream visualization construction. It includes paired direct-versus-staged runs, cross-family local-model comparisons, repeated-seed analyses, reranking ablations, four rendered case studies, and an external 105-case transfer study.

## Repository structure

- `experiment/`: experiment scripts, frozen designs, retained outputs, validation summaries, and analysis tables.
- `experiment/ext_v1_20260810/`: external-transfer design and source cases.
- `results/external_transfer/full_qwen/`: complete direct and staged Qwen outputs for the 105 external cases.
- `results/external_transfer/full_analysis/`: aggregate and paired external-transfer results.

## Environment

The machine experiments use local [Ollama](https://ollama.com/) models. Use Python 3.10 or newer. Package versions used for the analysis environment are recorded in `requirements.txt`; model names, prompts, seeds, and generation settings are retained with the run outputs and manifests.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the required Ollama models before rerunning machine inference. A short smoke run is recommended before a full campaign because model availability and runtime requirements vary by host.

## Reproducing analyses

The external-transfer analysis can be recomputed directly from the retained outputs:

```bash
python experiment/analyze_forward.py \
  --input-dir results/external_transfer/full_qwen \
  --output-dir reproduced/external_transfer \
  --data-dir experiment/ext_v1_20260810 \
  --split test \
  --bootstrap 2000
```

The command writes the same point estimates as `results/external_transfer/full_analysis/`; bootstrap limits vary only if the resampling count changes. Additional analysis entry points expose their options through `--help`, including `experiment/analyze_reranker.py` and `experiment/analyze_paired_conditions.py`. The complete retained outputs allow metric recomputation without rerunning model inference.

## Data and scope

The included benchmark-derived records contain only the fields required for the reported experiments. Paths in historical runtime manifests have been normalized to portable placeholders. This archive contains no manuscript source, submission file, credentials, or personally identifying participant data.

## Persistent archive

A versioned research snapshot is also available at [Zenodo](https://doi.org/10.5281/zenodo.21914363).

## License

Code is released under the MIT License. Benchmark-derived data remain subject to their original licenses and terms.
