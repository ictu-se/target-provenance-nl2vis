#!/usr/bin/env python3
"""Validate concrete workflow audits against the executable stage schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import Draft202012Validator


INSTANCES = [
    {
        "workflow_id": "internal_privileged_prototype",
        "input_contract": "violated",
        "candidate_ancestry": "partial",
        "target_containment": "measured",
        "forward_replacement": "different_mechanism",
        "output_layers": ["exact_target", "component_overlap"],
        "coverage_before_ordering": "measured",
        "claim_boundary": "retained forensic record; original constructor source and matched all-fields-removed run unavailable",
    },
    {
        "workflow_id": "public_nvbench2_sft_path",
        "input_contract": "verified",
        "candidate_ancestry": "source_only",
        "target_containment": "source_checked",
        "forward_replacement": "not_run",
        "output_layers": [],
        "coverage_before_ordering": "not_available",
        "claim_boundary": "source-level positive control; no trained-model score reproduced",
    },
    {
        "workflow_id": "locked_nvbench2_forward_and_reranking",
        "input_contract": "verified",
        "candidate_ancestry": "complete",
        "target_containment": "not_applicable",
        "forward_replacement": "different_mechanism",
        "output_layers": ["parseability", "execution", "benchmark_conformity", "exact_target", "component_overlap"],
        "coverage_before_ordering": "measured",
        "claim_boundary": "design-based inference to the fixed 720-record development population only",
    },
    {
        "workflow_id": "viseval_candidate_field_pair",
        "input_contract": "violated",
        "candidate_ancestry": "complete",
        "target_containment": "measured",
        "forward_replacement": "matched",
        "output_layers": ["parseability", "exact_target", "component_overlap"],
        "coverage_before_ordering": "single_candidate",
        "claim_boundary": "independent-benchmark diagnostic using retained five-model outputs; not a published pipeline reproduction",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    results = []
    for instance in INSTANCES:
        errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
        results.append(
            {
                "workflow_id": instance["workflow_id"],
                "valid": not errors,
                "errors": [error.message for error in errors],
                "record": instance,
            }
        )
    payload = {
        "status": "complete" if all(row["valid"] for row in results) else "failed",
        "schema": str(args.schema),
        "required_artifacts": list(schema["required"]),
        "instances": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
