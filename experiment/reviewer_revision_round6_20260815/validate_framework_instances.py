#!/usr/bin/env python3
"""Validate structured workflow audits and resolve their empirical evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_evidence(spec: dict[str, Any], root: Path) -> tuple[Any, list[str]]:
    path = (root / spec["path"]).resolve()
    errors: list[str] = []
    if not path.is_file():
        return None, [f"missing evidence file: {spec['path']}"]
    if spec["format"] == "csv":
        payload: Any = read_csv(path)
        fields = set(payload[0]) if payload else set()
        row_count = len(payload)
    elif spec["format"] == "jsonl":
        payload = read_jsonl(path)
        fields = set(payload[0]) if payload else set()
        row_count = len(payload)
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        fields = set(payload) if isinstance(payload, dict) else set()
        row_count = len(payload) if isinstance(payload, list) else None
    missing = set(spec["required_fields"]) - fields
    if missing:
        errors.append(f"{spec['evidence_id']} missing fields: {sorted(missing)}")
    if spec["min_rows"] is not None and (row_count is None or row_count < spec["min_rows"]):
        errors.append(f"{spec['evidence_id']} has {row_count} rows; expected at least {spec['min_rows']}")
    return payload, errors


def referenced_evidence_ids(record: dict[str, Any]) -> set[str]:
    refs = set(record["input_contract"]["evidence_ids"])
    ancestry_ref = record["candidate_ancestry"]["evidence_id"]
    containment_ref = record["target_containment"]["evidence_id"]
    if ancestry_ref:
        refs.add(ancestry_ref)
    if containment_ref:
        refs.add(containment_ref)
    for contrast in record["counterfactuals"]:
        refs.update(contrast["evidence_ids"])
    for layer in record["output_layers"]:
        refs.update(layer["evidence_ids"])
    refs.update(record["coverage_before_ordering"]["evidence_ids"])
    refs.update(record["ranking"]["evidence_ids"])
    return refs


def check_stage_graph(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for stage in record["stage_graph"]:
        stage_id = stage["stage_id"]
        if stage_id in seen:
            errors.append(f"duplicate stage_id: {stage_id}")
        unknown = set(stage["depends_on"]) - seen
        if unknown:
            errors.append(f"stage {stage_id} depends on unknown or later stages: {sorted(unknown)}")
        seen.add(stage_id)
    return errors


def filtered_rows(rows: list[dict[str, Any]], case_field: str | None, prefix: str | None) -> list[dict[str, Any]]:
    if not prefix or not case_field:
        return rows
    return [row for row in rows if str(row.get(case_field, "")).startswith(prefix)]


def check_ancestry(record: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    spec = record["candidate_ancestry"]
    if not spec["evidence_id"]:
        return []
    payload = evidence[spec["evidence_id"]]
    checks = set(spec["checks"])
    errors: list[str] = []
    if "public_prompt_separation" in checks:
        if not isinstance(payload, dict):
            return ["public prompt-separation evidence must be a JSON object"]
        if payload.get("gold_answer_injected_into_inference_prompt") is not False:
            errors.append("public prompt separation failed: gold answer is not confirmed absent")
        if payload.get("per_instance_target_exposure_found") is not False:
            errors.append("public prompt separation failed: target exposure reported")
        return errors
    if not isinstance(payload, list):
        return ["ancestry evidence must be row-oriented for the registered checks"]
    case_field = spec["case_id_field"]
    rows = filtered_rows(payload, case_field, spec["record_prefix"])
    if "expected_cases" in checks:
        unique_cases = {str(row[case_field]) for row in rows}
        if len(unique_cases) != spec["expected_case_count"]:
            errors.append(f"found {len(unique_cases)} cases; expected {spec['expected_case_count']}")
    if "unique_case_ids" in checks:
        ids = [str(row[case_field]) for row in rows]
        if len(ids) != len(set(ids)):
            errors.append("case identifiers are not unique in ancestry evidence")
    if "gold_fields_exposed_all" in checks:
        if any(int(float(row["gold_fields_exposed"])) != 1 for row in rows):
            errors.append("not every privileged VisEval row exposes both gold fields")
    if {"pool_count_matches", "sources_nonempty", "ranked_ids_resolve"} & checks:
        for row_number, row in enumerate(rows, start=1):
            details = row.get("pool_details", [])
            declared_pool_count = int(row.get("eligible_pool_count", len(details)))
            if "pool_count_matches" in checks and len(details) != declared_pool_count:
                errors.append(f"row {row_number}: pool_details count mismatch")
            if "sources_nonempty" in checks and any(not item.get("sources") for item in details):
                errors.append(f"row {row_number}: candidate without retained source ancestry")
            if "ranked_ids_resolve" in checks:
                valid = {f"c{index}" for index in range(len(details))}
                unresolved = set(row.get("llm_ranked_ids", [])) - valid
                if unresolved:
                    errors.append(f"row {row_number}: unresolved ranked IDs {sorted(unresolved)}")
    return errors


def check_containment(record: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    spec = record["target_containment"]
    if spec["status"] != "measured":
        return []
    payload = evidence[spec["evidence_id"]]
    if not isinstance(payload, list):
        return ["measured containment requires row-oriented evidence"]
    errors: list[str] = []
    for metric in spec["metrics"]:
        case_field = record["candidate_ancestry"]["case_id_field"]
        rows = filtered_rows(payload, case_field, metric["record_prefix"])
        values = [int(float(row[metric["column"]])) for row in rows]
        numerator, denominator = sum(values), len(values)
        rate = numerator / denominator if denominator else math.nan
        if numerator != metric["numerator"] or denominator != metric["denominator"]:
            errors.append(f"{metric['stage_id']}: declared {metric['numerator']}/{metric['denominator']} but evidence gives {numerator}/{denominator}")
        if not math.isclose(rate, metric["rate"], rel_tol=0, abs_tol=1e-12):
            errors.append(f"{metric['stage_id']}: declared rate {metric['rate']} but evidence gives {rate}")
    return errors


def validate_record(record: dict[str, Any], schema: dict[str, Any], root: Path) -> dict[str, Any]:
    errors = [error.message for error in sorted(Draft202012Validator(schema).iter_errors(record), key=lambda item: list(item.path))]
    if errors:
        return {"workflow_id": record.get("workflow_id", "unknown"), "valid": False, "errors": errors, "checks": {}}
    evidence_specs = {item["evidence_id"]: item for item in record["evidence"]}
    if len(evidence_specs) != len(record["evidence"]):
        errors.append("duplicate evidence_id")
    unknown_refs = referenced_evidence_ids(record) - set(evidence_specs)
    if unknown_refs:
        errors.append(f"unknown evidence references: {sorted(unknown_refs)}")
    evidence_payloads: dict[str, Any] = {}
    for evidence_id, spec in evidence_specs.items():
        payload, evidence_errors = load_evidence(spec, root)
        evidence_payloads[evidence_id] = payload
        errors.extend(evidence_errors)
    observed = set(record["input_contract"]["observed_fields"])
    privileged = set(record["input_contract"]["privileged_fields"])
    overlap = observed & privileged
    status = record["input_contract"]["status"]
    if status == "verified" and overlap:
        errors.append(f"verified input contract observes privileged fields: {sorted(overlap)}")
    if status == "violated" and not overlap:
        errors.append("violated input contract has no observed privileged field")
    stage_errors = check_stage_graph(record)
    errors.extend(stage_errors)
    ancestry_errors = check_ancestry(record, evidence_payloads) if not unknown_refs else []
    containment_errors = check_containment(record, evidence_payloads) if not unknown_refs else []
    errors.extend(ancestry_errors)
    errors.extend(containment_errors)
    checks = {
        "schema": "pass",
        "evidence_resolution": "pass" if not any("evidence" in error or "missing fields" in error or "rows" in error for error in errors) else "fail",
        "input_contract_invariant": "pass" if not any("input contract" in error for error in errors) else "fail",
        "stage_graph_invariant": "pass" if not stage_errors else "fail",
        "candidate_ancestry_invariants": "pass" if not ancestry_errors else "fail",
        "containment_recomputation": "pass" if not containment_errors else "fail"
    }
    return {"workflow_id": record["workflow_id"], "valid": not errors, "errors": errors, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    records = json.loads(args.records.read_text(encoding="utf-8"))
    root = args.records.parent
    results = [validate_record(record, schema, root) for record in records]
    payload = {
        "status": "complete" if all(row["valid"] for row in results) else "failed",
        "schema_version": "2.0.0",
        "records": len(records),
        "validation_layers": ["JSON Schema structure", "evidence resolution", "input-contract invariant", "stage-graph invariant", "candidate-ancestry invariant", "containment recomputation"],
        "instances": results
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
