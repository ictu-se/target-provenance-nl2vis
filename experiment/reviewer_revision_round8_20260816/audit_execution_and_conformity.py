#!/usr/bin/env python3
"""Audit execution before/after type completion and checker agreement.

The candidate is always attached to the same source rows, Vega-Lite schema URL,
and canvas size. The source-attached condition preserves every declared field
type and does not insert a missing type. The completed condition additionally
inserts a deterministic type only where the candidate omitted it.

The reference normal-form checker below is an independent implementation: it
does not import or call the production validation function and does not read its
saved error labels until after computing a reference label.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import vl_convert as vlc


ALLOWED_MARKS = {"bar", "line", "arc", "point", "rect", "boxplot"}
ALLOWED_CHANNELS = {"x", "y", "theta", "color", "size", "row", "column", "detail"}
REQUIRED_CHANNELS = {
    "bar": {"x", "y"},
    "line": {"x", "y"},
    "arc": {"color", "theta"},
    "point": {"x", "y"},
    "rect": {"x", "y", "color"},
    "boxplot": {"x", "y"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def coerce(value: str) -> Any:
    if value == "":
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", value):
        return float(value)
    return value


def load_values(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: coerce(value) for key, value in row.items()})
            if len(rows) >= limit:
                break
    return rows


def inferred_type(field: Any, values: list[dict[str, Any]], definition: dict[str, Any]) -> str:
    if definition.get("aggregate"):
        return "quantitative"
    if not isinstance(field, str):
        return "nominal"
    observed = [row.get(field) for row in values if row.get(field) is not None]
    if observed and all(isinstance(value, (int, float)) for value in observed):
        return "quantitative"
    if observed and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)) for value in observed):
        return "temporal"
    return "nominal"


def source_attached_spec(spec: dict[str, Any], values: list[dict[str, Any]], complete_types: bool) -> dict[str, Any]:
    output = copy.deepcopy(spec)
    if complete_types:
        for definition in output.get("encoding", {}).values():
            if isinstance(definition, dict) and not definition.get("type"):
                definition["type"] = inferred_type(definition.get("field"), values, definition)
    output.update({
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": values},
        "width": 120,
        "height": 80,
    })
    return output


def executes(spec: dict[str, Any]) -> tuple[int, str]:
    try:
        vlc.vegalite_to_svg(spec)
        return 1, ""
    except Exception as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def reference_errors(spec: dict[str, Any], columns: set[str]) -> list[str]:
    """Independent direct implementation of the registered normal form."""
    errors: set[str] = set()
    raw_mark = spec.get("mark")
    mark = raw_mark.get("type") if isinstance(raw_mark, dict) else raw_mark
    mark = str(mark).strip().lower() if isinstance(mark, str) else ""
    if mark not in ALLOWED_MARKS:
        errors.add("invalid_mark")

    encoding = spec.get("encoding")
    if not isinstance(encoding, dict):
        return sorted(errors | {"missing_encoding"})
    required = REQUIRED_CHANNELS.get(mark, set())
    if not required.issubset(set(encoding)):
        errors.add("missing_required_channel")
    for channel, definition in encoding.items():
        if channel not in ALLOWED_CHANNELS:
            errors.add(f"unsupported_channel:{channel}")
        if not isinstance(definition, dict):
            errors.add(f"non_object_channel:{channel}")
            continue
        field = definition.get("field")
        if isinstance(field, str) and field not in columns:
            errors.add(f"unknown_field:{field}")
        if definition.get("aggregate") == "count" and field is not None:
            errors.add(f"count_with_field:{channel}")

    transforms = spec.get("transform", [])
    if transforms is None:
        transforms = []
    if not isinstance(transforms, list):
        errors.add("unsupported_transform")
        transforms = []
    for transform in transforms:
        if not isinstance(transform, dict) or "filter" not in transform:
            errors.add("unsupported_transform")
            continue
        filt = transform.get("filter")
        if not isinstance(filt, dict):
            errors.add("non_object_filter")
        elif isinstance(filt.get("field"), str) and filt["field"] not in columns:
            errors.add(f"unknown_filter_field:{filt['field']}")
    return sorted(errors)


def weighted_mean(cases: list[dict[str, Any]], populations: dict[str, int], metric: str) -> float:
    strata: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        strata[str(case["gold_family"])].append(float(case[metric]))
    total = sum(populations.values())
    return sum(populations[name] * mean(strata[name]) for name in populations) / total


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--dev-json", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    design = json.loads(args.design.read_text(encoding="utf-8"))
    populations = {str(k): int(v) for k, v in design["population_strata_after_screen"].items()}
    selected = {int(value) for value in design["indices"]}
    dev_rows = json.loads(args.dev_json.read_text(encoding="utf-8"))
    records = {
        int(row.get("index", index)): row
        for index, row in enumerate(dev_rows)
        if int(row.get("index", index)) in selected
    }

    per_candidate: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    error_confusion: Counter[tuple[str, str]] = Counter()
    for path in sorted(args.input_dir.glob("*.jsonl")):
        for run in read_jsonl(path):
            index = int(run["index"])
            record = records[index]
            values = load_values(args.csv_dir / record["csv_file"])
            table_schema = record["table_schema"]
            if isinstance(table_schema, str):
                table_schema = json.loads(table_schema)
            columns = {str(item) for item in table_schema["table_columns"]}
            candidates = run.get("candidates", [])
            production_labels = run.get("candidate_validation_errors", [])
            source_flags: list[int] = []
            completed_flags: list[int] = []
            for rank, candidate in enumerate(candidates, start=1):
                source_ok, source_error = executes(source_attached_spec(candidate, values, False))
                completed_ok, completed_error = executes(source_attached_spec(candidate, values, True))
                reference = reference_errors(candidate, columns)
                production = sorted(set(production_labels[rank - 1])) if rank - 1 < len(production_labels) else ["missing_label"]
                reference_accept = int(not reference)
                production_accept = int(not production)
                error_confusion[(str(reference), str(production))] += 1
                source_flags.append(source_ok)
                completed_flags.append(completed_ok)
                per_candidate.append({
                    "record_id": run["record_id"],
                    "model": run["model"],
                    "condition": run["condition"],
                    "rank": rank,
                    "source_attached_execution": source_ok,
                    "type_completed_execution": completed_ok,
                    "completion_rescue": int(not source_ok and completed_ok),
                    "completion_break": int(source_ok and not completed_ok),
                    "production_accept": production_accept,
                    "reference_accept": reference_accept,
                    "accept_agreement": int(production_accept == reference_accept),
                    "error_set_agreement": int(production == reference),
                    "production_errors": "|".join(production),
                    "reference_errors": "|".join(reference),
                    "source_error": source_error,
                    "completed_error": completed_error,
                })
            gold_answer = record["gold_answer"]
            if isinstance(gold_answer, str):
                gold_answer = json.loads(gold_answer)
            gold_family = "+".join(sorted({str(spec.get("mark")) for spec in gold_answer}))
            per_case.append({
                "record_id": run["record_id"],
                "model": run["model"],
                "condition": run["condition"],
                "gold_family": gold_family,
                "top1_source_attached_execution": source_flags[0] if source_flags else 0,
                "source_attached_execution_fraction": mean(source_flags) if source_flags else 0.0,
                "top1_type_completed_execution": completed_flags[0] if completed_flags else 0,
                "type_completed_execution_fraction": mean(completed_flags) if completed_flags else 0.0,
                "top1_completion_rescue": int(bool(source_flags) and not source_flags[0] and completed_flags[0]),
                "completion_rescue_fraction": mean(int(not a and b) for a, b in zip(source_flags, completed_flags)) if source_flags else 0.0,
            })

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in per_case:
        grouped[(str(case["model"]), str(case["condition"]))].append(case)
    metrics = [
        "top1_source_attached_execution", "source_attached_execution_fraction",
        "top1_type_completed_execution", "type_completed_execution_fraction",
        "top1_completion_rescue", "completion_rescue_fraction",
    ]
    summaries = []
    for (model, condition), cases in sorted(grouped.items()):
        if len(cases) != 150:
            raise SystemExit(f"{model}/{condition}: expected 150 cases, found {len(cases)}")
        row: dict[str, Any] = {"model": model, "condition": condition, "n": len(cases)}
        for metric in metrics:
            row[metric] = weighted_mean(cases, populations, metric)
        summaries.append(row)

    tp = sum(r["production_accept"] and r["reference_accept"] for r in per_candidate)
    fp = sum(r["production_accept"] and not r["reference_accept"] for r in per_candidate)
    fn = sum(not r["production_accept"] and r["reference_accept"] for r in per_candidate)
    tn = sum(not r["production_accept"] and not r["reference_accept"] for r in per_candidate)
    checker = [{
        "generated_candidates": len(per_candidate),
        "reference_accepts": tp + fn,
        "reference_rejects": tn + fp,
        "true_accept": tp,
        "false_accept": fp,
        "true_reject": tn,
        "false_reject": fn,
        "accept_precision": tp / (tp + fp) if tp + fp else 0.0,
        "accept_recall": tp / (tp + fn) if tp + fn else 0.0,
        "accept_agreement": (tp + tn) / len(per_candidate),
        "error_set_agreement": mean(r["error_set_agreement"] for r in per_candidate),
    }]

    write_csv(args.output_dir / "condition_summary.csv", summaries)
    write_csv(args.output_dir / "per_case.csv", per_case)
    write_csv(args.output_dir / "per_candidate.csv", per_candidate)
    write_csv(args.output_dir / "checker_agreement.csv", checker)
    manifest = {
        "status": "complete",
        "scope": "all retained candidates from 1,800 locked development model-condition rows",
        "candidate_rows": len(per_candidate),
        "source_attached_definition": "candidate plus source rows, schema URL, width, and height; no missing type is inserted",
        "type_completed_definition": "source-attached candidate plus deterministic insertion of missing encoding types",
        "renderer": f"vl-convert-python {getattr(vlc, '__version__', 'unknown')}",
        "reference_checker": "independent reimplementation; no import of production validation function",
        "checker_scope_boundary": "implementation agreement with declared benchmark normal form, not external expert semantic or perceptual validity",
        "source_attached_failures": sum(not r["source_attached_execution"] for r in per_candidate),
        "type_completed_failures": sum(not r["type_completed_execution"] for r in per_candidate),
        "completion_rescues": sum(r["completion_rescue"] for r in per_candidate),
        "completion_breaks": sum(r["completion_break"] for r in per_candidate),
        "checker": checker[0],
        "distinct_error_set_pairs": len(error_confusion),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
