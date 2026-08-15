# Round-6 provenance and robustness analyses

This directory contains the executable audit schema and analysis entry points
added after the final significance/generality review. Machine outputs and
derived results are under `results/reviewer_revision_round6_20260815/`.

## Evidence blocks

- `stage_audit_schema.json` requires an input contract, candidate ancestry,
  target-containment status, forward-replacement status, output layers,
  coverage-before-ordering status, and an explicit claim boundary.
- `analyze_viseval_provenance.py` pairs query-only and deliberately privileged
  VisEval conditions. The privileged condition contains candidate fields
  derived from released gold SQL and normalized gold x/y fields.
- `run_prompt_schema_robustness.py` reruns Qwen direct-rich on the locked 150
  development cases under a semantic prompt paraphrase and reversed schema
  column order. The protocol was fixed before these two runs.
- `analyze_prompt_schema_robustness.py` reports candidate identity, set
  Jaccard, exact retrieval, benchmark conformity, and component overlap.

## Recompute deterministic analyses

Run these commands from the repository root:

```bash
python experiment/reviewer_revision_round6_20260815/validate_framework_instances.py \
  --schema experiment/reviewer_revision_round6_20260815/design/stage_audit_schema.json \
  --output results/reviewer_revision_round6_20260815/analysis/framework_validation.json

python experiment/reviewer_revision_round6_20260815/analyze_viseval_provenance.py \
  --query-only results/reviewer_revision_round6_20260815/input/viseval_query_only.jsonl \
  --privileged results/reviewer_revision_round6_20260815/input/viseval_privileged_fields.jsonl \
  --output-dir results/reviewer_revision_round6_20260815/analysis/viseval_provenance

python experiment/reviewer_revision_round6_20260815/analyze_prompt_schema_robustness.py \
  --baseline results/reviewer_revision_round2_20260814/runs/holdout/qwen3_14b_613573_r_55_d_r2holdout150.jsonl \
  --variant-dir results/reviewer_revision_round6_20260815/runs/robustness \
  --data-dir PATH_TO_NVBench_2_DATA \
  --design experiment/reviewer_revision_round2_20260814/design/dev_locked_holdout150.json \
  --output-dir results/reviewer_revision_round6_20260815/analysis/prompt_schema_robustness
```

The VisEval inputs contain retained rows from two public-benchmark campaigns;
the analysis selects `benchmark=viseval`, yielding 100 cases for each of five
model families in each condition. These analyses diagnose provenance and
configuration sensitivity. They do not reproduce a published VisEval system,
validate semantic equivalence, or support future-population inference.
