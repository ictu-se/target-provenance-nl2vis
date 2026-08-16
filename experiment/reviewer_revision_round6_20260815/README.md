# Reviewer revision round 6

This directory contains the independent VisEval provenance diagnostic, prompt/schema robustness audit, and version-2 executable stage-audit contract.

The stage audit is not a checklist of self-declared statuses. Each workflow record binds structured input contracts, stage graphs, candidate ancestry, containment measurements, output layers, coverage, ranking, and claim boundaries to resolvable empirical artifacts. The validator checks schema structure, evidence files and fields, input-contract consistency, stage dependencies, candidate IDs and sources, and registered containment arithmetic.

Run the positive validation from this directory:

```bash
python3 validate_framework_instances.py \
  --schema design/stage_audit_schema.json \
  --records design/workflow_audit_records.json \
  --output ../../../results/reviewer_revision_round6_20260815/analysis/framework_validation.json
```

Run the registered negative controls:

```bash
python3 test_stage_audit_validator.py \
  --schema design/stage_audit_schema.json \
  --records design/workflow_audit_records.json \
  --output ../../../results/reviewer_revision_round6_20260815/analysis/framework_negative_controls.json
```

The negative controls remove a required claim boundary, mislabel a violated input contract, insert an invalid stage dependency, remove evidence, alter a containment numerator, remove candidate ancestry, and insert an unresolved ranked ID. All seven mutations must be rejected.
