#!/usr/bin/env python3
"""Negative controls for the executable stage-audit validator."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from validate_framework_instances import check_ancestry, validate_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    records = json.loads(args.records.read_text(encoding="utf-8"))
    root = args.records.parent

    mutations: list[dict[str, Any]] = []

    missing_structure = copy.deepcopy(records[0])
    missing_structure.pop("claim_boundary")
    result = validate_record(missing_structure, schema, root)
    mutations.append({"mutation": "remove_required_claim_boundary", "rejected": not result["valid"], "errors": result["errors"]})

    wrong_contract = copy.deepcopy(records[0])
    wrong_contract["input_contract"]["status"] = "verified"
    result = validate_record(wrong_contract, schema, root)
    mutations.append({"mutation": "label_privileged_inputs_verified", "rejected": not result["valid"], "errors": result["errors"]})

    broken_graph = copy.deepcopy(records[1])
    broken_graph["stage_graph"][1]["depends_on"] = ["future_stage"]
    result = validate_record(broken_graph, schema, root)
    mutations.append({"mutation": "insert_unknown_stage_dependency", "rejected": not result["valid"], "errors": result["errors"]})

    missing_evidence = copy.deepcopy(records[3])
    missing_evidence["evidence"][0]["path"] = "missing/paired_rows.csv"
    result = validate_record(missing_evidence, schema, root)
    mutations.append({"mutation": "remove_empirical_evidence", "rejected": not result["valid"], "errors": result["errors"]})

    wrong_containment = copy.deepcopy(records[0])
    wrong_containment["target_containment"]["metrics"][0]["numerator"] += 1
    result = validate_record(wrong_containment, schema, root)
    mutations.append({"mutation": "alter_containment_numerator", "rejected": not result["valid"], "errors": result["errors"]})

    locked = records[2]
    evidence_path = (root / locked["evidence"][0]["path"]).resolve()
    rows = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

    empty_source_rows = copy.deepcopy(rows)
    empty_source_rows[0]["pool_details"][0]["sources"] = []
    errors = check_ancestry(locked, {"locked_rerank_jsonl": empty_source_rows})
    mutations.append({"mutation": "remove_candidate_source_ancestry", "rejected": any("without retained source" in error for error in errors), "errors": errors})

    unresolved_rows = copy.deepcopy(rows)
    unresolved_rows[0]["llm_ranked_ids"][0] = "c999"
    errors = check_ancestry(locked, {"locked_rerank_jsonl": unresolved_rows})
    mutations.append({"mutation": "insert_unresolved_ranked_id", "rejected": any("unresolved ranked IDs" in error for error in errors), "errors": errors})

    payload = {
        "status": "complete" if all(item["rejected"] for item in mutations) else "failed",
        "registered_mutations": len(mutations),
        "rejected_mutations": sum(item["rejected"] for item in mutations),
        "mutations": mutations
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

