#!/usr/bin/env python3
"""Compare prompt/schema perturbations with the retained Qwen direct-rich run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
ROUND2 = HERE.parent / "reviewer_revision_round2_20260814"
BASE = HERE.parent / "reviewer_revision_20260807"
sys.path.insert(0, str(ROUND2))
sys.path.insert(0, str(BASE))

from analyze_forward import evaluate_case  # noqa: E402
from analyze_locked_holdout import bootstrap_ci, weighted_mean  # noqa: E402
from common import canonical_key, dedupe_specs, load_split  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_keys(row: dict[str, Any]) -> list[str]:
    return [key for item in dedupe_specs(row.get("candidates", []))[:5] if (key := canonical_key(item))]


def jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    design = json.loads(args.design.read_text(encoding="utf-8"))
    populations = {str(key): int(value) for key, value in design["population_strata_after_screen"].items()}
    selected = {int(index) for index in design["indices"]}
    gold = {
        row["record_id"]: row
        for row in load_split(args.data_dir, "dev")
        if int(row["index"]) in selected
    }
    baseline = {str(row["record_id"]): row for row in read_jsonl(args.baseline)}
    if set(baseline) != set(gold):
        raise SystemExit("baseline does not match the locked 150-case design")

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for path in sorted(args.variant_dir.glob("*.jsonl")):
        variants = {str(row["record_id"]): row for row in read_jsonl(path)}
        if set(variants) != set(gold):
            raise SystemExit(f"{path.name} does not match locked cases: {len(variants)}")
        cases: list[dict[str, Any]] = []
        variant_name = str(next(iter(variants.values()))["robustness_variant"])
        for record_id in sorted(gold):
            base_row, variant_row, gold_row = baseline[record_id], variants[record_id], gold[record_id]
            base_keys, variant_keys = candidate_keys(base_row), candidate_keys(variant_row)
            base_eval, variant_eval = evaluate_case(base_row, gold_row), evaluate_case(variant_row, gold_row)
            base_errors = base_row.get("candidate_validation_errors", [])
            variant_errors = variant_row.get("candidate_validation_errors", [])
            base_conformity = float(bool(base_keys) and bool(base_errors) and not base_errors[0])
            variant_conformity = float(bool(variant_keys) and bool(variant_errors) and not variant_errors[0])
            case = {
                "record_id": record_id,
                "gold_family": base_eval["gold_family"],
                "variant": variant_name,
                "top1_agreement": float(bool(base_keys) and bool(variant_keys) and base_keys[0] == variant_keys[0]),
                "candidate_jaccard": jaccard(base_keys, variant_keys),
                "baseline_hit1": float(base_eval["raw_hit@1"]),
                "variant_hit1": float(variant_eval["raw_hit@1"]),
                "delta_hit1": float(variant_eval["raw_hit@1"] - base_eval["raw_hit@1"]),
                "baseline_hit5": float(base_eval["raw_hit@5"]),
                "variant_hit5": float(variant_eval["raw_hit@5"]),
                "delta_hit5": float(variant_eval["raw_hit@5"] - base_eval["raw_hit@5"]),
                "baseline_top1_conformity": base_conformity,
                "variant_top1_conformity": variant_conformity,
                "delta_top1_conformity": variant_conformity - base_conformity,
                "baseline_component_overlap": float(base_eval["top1_macro"]),
                "variant_component_overlap": float(variant_eval["top1_macro"]),
                "delta_component_overlap": float(variant_eval["top1_macro"] - base_eval["top1_macro"]),
                "variant_parse_success": float(bool(variant_row.get("parse_success"))),
            }
            cases.append(case)
            detail_rows.append(case)

        summary: dict[str, Any] = {"variant": variant_name, "n": len(cases)}
        for offset, metric in enumerate(
            ("top1_agreement", "candidate_jaccard", "baseline_hit1", "variant_hit1", "delta_hit1", "baseline_hit5", "variant_hit5", "delta_hit5", "baseline_top1_conformity", "variant_top1_conformity", "delta_top1_conformity", "baseline_component_overlap", "variant_component_overlap", "delta_component_overlap", "variant_parse_success")
        ):
            value = lambda row, name=metric: float(row[name])
            summary[metric] = weighted_mean(cases, populations, value)
            low, high = bootstrap_ci(cases, populations, value, args.bootstrap, args.seed + offset)
            summary[f"{metric}_ci_low"] = low
            summary[f"{metric}_ci_high"] = high
        summary_rows.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    fields = list(dict.fromkeys(key for row in summary_rows for key in row))
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    manifest = {
        "status": "complete",
        "variants": [row["variant"] for row in summary_rows],
        "sample": "same locked 150-case development sample",
        "model": "qwen3:14b",
        "condition": "direct-rich",
        "claim_boundary": "post-review local robustness sensitivity",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
