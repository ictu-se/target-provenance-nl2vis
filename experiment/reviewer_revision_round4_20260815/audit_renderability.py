#!/usr/bin/env python3
"""Separate response parsing, renderer execution, and registered compliance.

The audit uses the already locked 150-record development sample.  It does not
regenerate candidates.  Each retained candidate is compiled to SVG by the
Vega-Lite renderer after adding only renderer-required data, type declarations,
and canvas metadata.  Marks, fields, aggregates, transforms, and filters are
not changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import vl_convert as vlc


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


def inferred_type(field: str | None, values: list[dict[str, Any]], definition: dict[str, Any]) -> str:
    if definition.get("aggregate"):
        return "quantitative"
    observed = [row.get(field) for row in values if field and row.get(field) is not None]
    if observed and all(isinstance(value, (int, float)) for value in observed):
        return "quantitative"
    if observed and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)) for value in observed):
        return "temporal"
    return "nominal"


def renderer_spec(spec: dict[str, Any], values: list[dict[str, Any]]) -> dict[str, Any]:
    rendered = json.loads(json.dumps(spec))
    for definition in rendered.get("encoding", {}).values():
        if isinstance(definition, dict) and not definition.get("type"):
            definition["type"] = inferred_type(definition.get("field"), values, definition)
    rendered.update(
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": values},
            "width": 120,
            "height": 80,
        }
    )
    return rendered


def weighted_mean(cases: list[dict[str, Any]], populations: dict[str, int], metric: str) -> float:
    strata: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        strata[str(case["gold_family"])].append(float(case[metric]))
    total = sum(populations.values())
    return sum(populations[name] * mean(strata[name]) for name in populations) / total


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    records = {int(row.get("index", index)): row for index, row in enumerate(dev_rows) if int(row.get("index", index)) in selected}

    per_case: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    files = []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        files.append({"file": path.name, "rows": len(rows), "sha256": sha256(path)})
        for run in rows:
            index = int(run["index"])
            record = records[index]
            values = load_values(args.csv_dir / record["csv_file"])
            candidates = run.get("candidates", [])
            validation = run.get("candidate_validation_errors", [])
            flags = []
            for rank, candidate in enumerate(candidates, start=1):
                error = None
                try:
                    vlc.vegalite_to_svg(renderer_spec(candidate, values))
                    renderable = 1.0
                except Exception as exc:  # renderer diagnostics are part of the audit output
                    renderable = 0.0
                    error = f"{type(exc).__name__}: {exc}"
                compliant = float(rank - 1 < len(validation) and not validation[rank - 1])
                flags.append(renderable)
                candidate_rows.append(
                    {
                        "record_id": run["record_id"],
                        "model": run["model"],
                        "condition": run["condition"],
                        "rank": rank,
                        "renderer_executable": int(renderable),
                        "registered_compliant": int(compliant),
                        "registered_errors": "|".join(validation[rank - 1]) if rank - 1 < len(validation) else "missing_validation_record",
                        "renderer_error": error or "",
                    }
                )
            gold_answer = record["gold_answer"]
            if isinstance(gold_answer, str):
                gold_answer = json.loads(gold_answer)
            gold_family = "+".join(sorted({str(spec.get("mark")) for spec in gold_answer}))
            per_case.append(
                {
                    "record_id": run["record_id"],
                    "model": run["model"],
                    "condition": run["condition"],
                    "gold_family": gold_family,
                    "response_parseable": float(bool(run.get("parse_success"))),
                    "any_renderer_executable": float(any(flags)),
                    "top1_renderer_executable": flags[0] if flags else 0.0,
                    "renderer_executable_fraction": mean(flags) if flags else 0.0,
                    "top1_registered_compliant": float(bool(candidates) and not validation[0]),
                    "registered_compliant_fraction": mean(float(not errors) for errors in validation) if validation else 0.0,
                    "candidate_count": len(candidates),
                }
            )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in per_case:
        grouped[(str(case["model"]), str(case["condition"]))].append(case)
    metrics = [
        "response_parseable",
        "any_renderer_executable",
        "top1_renderer_executable",
        "renderer_executable_fraction",
        "top1_registered_compliant",
        "registered_compliant_fraction",
        "candidate_count",
    ]
    summaries = []
    for (model, condition), cases in sorted(grouped.items()):
        if len(cases) != 150:
            raise SystemExit(f"{model}/{condition}: expected 150 rows, found {len(cases)}")
        row: dict[str, Any] = {"model": model, "condition": condition, "n": len(cases)}
        for metric in metrics:
            row[metric] = weighted_mean(cases, populations, metric)
        summaries.append(row)

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = list(rows[0])
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "condition_summary.csv", summaries)
    write_csv(args.output_dir / "per_case.csv", per_case)
    write_csv(args.output_dir / "per_candidate.csv", candidate_rows)
    manifest = {
        "status": "complete",
        "scope": "locked 150-record nvBench-2.0 development sample; retained outputs only",
        "renderer": f"vl-convert-python {getattr(vlc, '__version__', 'unknown')}",
        "renderer_test": "Vega-Lite-to-SVG execution with at most 200 source rows per chart",
        "allowed_additions": "data values, inferred encoding type when absent, schema URL, width, and height",
        "prohibited_changes": "marks, fields, aggregates, transforms, filters, or candidate order",
        "files": files,
        "run_rows": len(per_case),
        "candidate_rows": len(candidate_rows),
        "renderer_failures": sum(not row["renderer_executable"] for row in candidate_rows),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
