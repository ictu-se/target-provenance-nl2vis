# Reviewer revision round 8

This round strengthens three measurement checks on the already locked
150-record development study. It performs no new model inference.

- `audit_execution_and_conformity.py` separates source-attached execution with
  the candidate's declared field types from execution after deterministic type
  completion. It also reimplements the registered normal-form rules without
  importing the production checker and compares both implementations on all
  retained generated candidates.
- `validate_representation_alignment.py` inspects every primary staged-rich
  exact-gain case previously classified as having identical mark, channel,
  field, operation, and filter sets. It compares the direct-rich baseline with
  the matched gold after deterministic type completion and SVG rendering.

The reference checker validates implementation agreement with the declared
normal form. It is not an external expert judgment of analytical, perceptual,
or semantic chart quality.
