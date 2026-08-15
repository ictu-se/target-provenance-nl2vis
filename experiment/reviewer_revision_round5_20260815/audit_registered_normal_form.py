#!/usr/bin/env python3
"""Validate the study-registered benchmark-normal-form checker."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
sys.path.insert(0, str(BASE))

from common import dedupe_specs  # noqa: E402
from run_forward_ollama import load_split, schema_columns, validation_errors  # noqa: E402


REQUIRED = {
    "bar": ["x", "y"], "line": ["x", "y"], "arc": ["color", "theta"],
    "point": ["x", "y"], "rect": ["x", "y", "color"], "boxplot": ["x", "y"],
}


def first_channel(spec: dict[str, Any]) -> str:
    encoding = spec.get("encoding", {})
    return next(iter(encoding), "x") if isinstance(encoding, dict) else "x"


def perturb(spec: dict[str, Any], family: str, columns: set[str]) -> tuple[dict[str, Any], str]:
    output = copy.deepcopy(spec)
    output.setdefault("encoding", {})
    channel = first_channel(output)
    known = sorted(columns)[0]
    if family == "invalid_mark":
        output["mark"] = "radar"
        target = "invalid_mark"
    elif family == "unsupported_channel":
        output["encoding"]["radius"] = {"field": known}
        target = "unsupported_channel:radius"
    elif family == "missing_required_channel":
        required = REQUIRED[str(output.get("mark"))]
        output["encoding"].pop(required[0], None)
        target = "missing_required_channel"
    elif family == "unknown_field":
        output["encoding"][channel] = {"field": "__not_in_schema__"}
        target = "unknown_field:__not_in_schema__"
    elif family == "field_bearing_count":
        output["encoding"][channel] = {"aggregate": "count", "field": known}
        target = f"count_with_field:{channel}"
    elif family == "unsupported_transform":
        output.setdefault("transform", []).append({"calculate": "1", "as": "x"})
        target = "unsupported_transform"
    elif family == "non_object_filter":
        output.setdefault("transform", []).append({"filter": "bad"})
        target = "non_object_filter"
    else:
        raise ValueError(family)
    return output, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    families = [
        "invalid_mark", "unsupported_channel", "missing_required_channel", "unknown_field",
        "field_bearing_count", "unsupported_transform", "non_object_filter",
    ]
    gold_total = 0
    gold_false_reject = 0
    rows = []
    error_counts: Counter[str] = Counter()
    for row in load_split(args.data_dir, "test"):
        columns = schema_columns(row)
        for gold in dedupe_specs(row["gold_answer"]):
            gold_total += 1
            if validation_errors(gold, columns):
                gold_false_reject += 1
            for family in families:
                altered, target = perturb(gold, family, columns)
                errors = validation_errors(altered, columns)
                detected = target in errors
                error_counts.update(errors)
                rows.append({
                    "record_id": row["record_id"], "family": family, "target_error": target,
                    "detected_target_error": int(detected), "rejected": int(bool(errors)),
                    "errors": "|".join(errors),
                })
    summary = []
    for family in families:
        group = [row for row in rows if row["family"] == family]
        summary.append({
            "perturbation": family, "n": len(group),
            "target_detection": sum(row["detected_target_error"] for row in group),
            "rejected": sum(row["rejected"] for row in group),
            "false_acceptance": sum(not row["rejected"] for row in group),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "perturbation_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    with (args.output_dir / "per_perturbation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {
        "status": "complete", "gold_specifications": gold_total,
        "gold_false_rejections": gold_false_reject, "gold_acceptance": gold_total - gold_false_reject,
        "negative_perturbations": len(rows), "perturbation_families": len(families),
        "false_acceptances": sum(not row["rejected"] for row in rows),
        "target_error_misses": sum(not row["detected_target_error"] for row in rows),
        "nonexclusive_error_counts": dict(error_counts),
        "scope": "internal validation of the study-registered benchmark-normal-form rules; not Vega-Lite, semantic, or perceptual validation",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
