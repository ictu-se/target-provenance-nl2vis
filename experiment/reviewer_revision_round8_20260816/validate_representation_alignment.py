#!/usr/bin/env python3
"""Validate all primary exact gains with component-identical baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import vl_convert as vlc

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_round8_20260816"
sys.path.insert(0, str(BASE))

from audit_execution_and_conformity import load_values, source_attached_spec  # noqa: E402


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def svg_hash(svg: str) -> str:
    return hashlib.sha256(svg.encode("utf-8")).hexdigest()


def byte_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--dev-json", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.taxonomy.open(encoding="utf-8-sig", newline="") as handle:
        taxonomy = list(csv.DictReader(handle))
    selected = [
        row for row in taxonomy
        if row["contrast"] == "staged-rich minus direct-rich"
        and row["baseline_difference"] == "other_representation_difference"
    ]
    dev_rows = json.loads(args.dev_json.read_text(encoding="utf-8"))
    records = {str(row.get("record_id", f"nvbench2:dev:{i}")): row for i, row in enumerate(dev_rows)}

    results: list[dict[str, Any]] = []
    for row in selected:
        record = records[row["record_id"]]
        values = load_values(args.csv_dir / record["csv_file"])
        baseline = json.loads(row["baseline_top1"])
        golds = json.loads(row["golds"])
        comparator = json.loads(row["comparator_top1"])
        matched = next(gold for gold in golds if stable(gold) == stable(comparator))
        baseline_completed = source_attached_spec(baseline, values, True)
        gold_completed = source_attached_spec(matched, values, True)
        baseline_svg = vlc.vegalite_to_svg(baseline_completed)
        gold_svg = vlc.vegalite_to_svg(gold_completed)
        baseline_png = vlc.vegalite_to_png(baseline_completed, scale=1)
        gold_png = vlc.vegalite_to_png(gold_completed, scale=1)
        completed_spec_equal = int(stable(baseline_completed) == stable(gold_completed))
        rendered_svg_equal = int(baseline_svg == gold_svg)
        rendered_png_equal = int(baseline_png == gold_png)
        results.append({
            "model": row["model"],
            "record_id": row["record_id"],
            "gold_family": row["gold_family"],
            "completed_spec_equal": completed_spec_equal,
            "rendered_svg_equal": rendered_svg_equal,
            "rendered_png_equal": rendered_png_equal,
            "baseline_svg_sha256": svg_hash(baseline_svg),
            "gold_svg_sha256": svg_hash(gold_svg),
            "baseline_png_sha256": byte_hash(baseline_png),
            "gold_png_sha256": byte_hash(gold_png),
            "baseline_declared_types": stable({k: v.get("type") for k, v in baseline.get("encoding", {}).items()}),
            "completed_gold_types": stable({k: v.get("type") for k, v in gold_completed.get("encoding", {}).items()}),
        })

    if len(results) != 19:
        raise SystemExit(f"expected 19 component-identical primary gains, found {len(results)}")
    write_csv(args.output_dir / "per_case.csv", results)
    model_summary = []
    for model in sorted({row["model"] for row in results}):
        group = [row for row in results if row["model"] == model]
        model_summary.append({
            "model": model,
            "cases": len(group),
            "completed_spec_equal": sum(row["completed_spec_equal"] for row in group),
            "rendered_svg_equal": sum(row["rendered_svg_equal"] for row in group),
            "rendered_png_equal": sum(row["rendered_png_equal"] for row in group),
        })
    write_csv(args.output_dir / "model_summary.csv", model_summary)
    result = {
        "status": "complete",
        "scope": "all staged-rich minus direct-rich exact gains whose baseline shares all five registered component sets with the matched gold",
        "cases": len(results),
        "models": dict(Counter(row["model"] for row in results)),
        "completed_spec_equal": sum(row["completed_spec_equal"] for row in results),
        "rendered_svg_equal": sum(row["rendered_svg_equal"] for row in results),
        "rendered_png_equal": sum(row["rendered_png_equal"] for row in results),
        "interpretation": "completed-spec/pixel equality supports execution-level representation equivalence only; unequal cases are not labelled analytically equivalent",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
