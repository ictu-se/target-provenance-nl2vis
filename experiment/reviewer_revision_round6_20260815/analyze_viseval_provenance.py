#!/usr/bin/env python3
"""Paired provenance audit on the independent VisEval benchmark.

The retained campaign compares a query-only prompt with a prompt that appends
candidate fields extracted from released SQL and gold field metadata.  The
latter is deliberately privileged and is analysed as a diagnostic rather than
as a deployment score.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ("json_valid", "chart_match", "field_pair_match", "full_match")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_delta(
    rows: list[dict[str, Any]], metric: str, repetitions: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    deltas = [float(row[f"privileged_{metric}"]) - float(row[f"query_only_{metric}"]) for row in rows]
    estimates = [mean(rng.choices(deltas, k=len(deltas))) for _ in range(repetitions)]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-only", type=Path, required=True)
    parser.add_argument("--privileged", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    base = {
        (str(row["case_id"]), str(row["model"])): row
        for row in read_jsonl(args.query_only)
        if row.get("benchmark") == "viseval"
    }
    privileged = {
        (str(row["case_id"]), str(row["model"])): row
        for row in read_jsonl(args.privileged)
        if row.get("benchmark") == "viseval"
    }
    if set(base) != set(privileged):
        raise SystemExit("VisEval paired keys differ between retained campaigns")

    paired: list[dict[str, Any]] = []
    for key in sorted(base):
        left, right = base[key], privileged[key]
        if left.get("query") != right.get("query"):
            raise SystemExit(f"query mismatch for {key}")
        fields = {str(value).lower() for value in right.get("candidate_fields", [])}
        gold_x, gold_y = str(right.get("gold_x", "")).lower(), str(right.get("gold_y", "")).lower()
        row: dict[str, Any] = {
            "case_id": key[0],
            "model": key[1],
            "gold_fields_exposed": int(bool(gold_x) and bool(gold_y) and gold_x in fields and gold_y in fields),
            "candidate_field_count": len(fields),
        }
        for metric in METRICS:
            row[f"query_only_{metric}"] = int(bool(left.get(metric)))
            row[f"privileged_{metric}"] = int(bool(right.get(metric)))
        paired.append(row)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        groups[str(row["model"])].append(row)

    summaries: list[dict[str, Any]] = []
    for model, rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "model": model,
            "cases": len(rows),
            "gold_fields_exposed_pct": 100 * mean(row["gold_fields_exposed"] for row in rows),
            "mean_candidate_field_count": mean(row["candidate_field_count"] for row in rows),
        }
        for offset, metric in enumerate(METRICS):
            q = mean(row[f"query_only_{metric}"] for row in rows)
            p = mean(row[f"privileged_{metric}"] for row in rows)
            low, high = bootstrap_delta(rows, metric, args.bootstrap, args.seed + offset)
            gains = sum(
                row[f"query_only_{metric}"] == 0 and row[f"privileged_{metric}"] == 1
                for row in rows
            )
            losses = sum(
                row[f"query_only_{metric}"] == 1 and row[f"privileged_{metric}"] == 0
                for row in rows
            )
            summary[f"query_only_{metric}_pct"] = 100 * q
            summary[f"privileged_{metric}_pct"] = 100 * p
            summary[f"delta_{metric}_pp"] = 100 * (p - q)
            summary[f"delta_{metric}_ci_low_pp"] = 100 * low
            summary[f"delta_{metric}_ci_high_pp"] = 100 * high
            summary[f"{metric}_gains"] = gains
            summary[f"{metric}_losses"] = losses
        summaries.append(summary)

    # Macro-average across the five fixed model families. Bootstrap cases and
    # preserve all model rows for a selected case in each replicate.
    case_ids = sorted({str(row["case_id"]) for row in paired})
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_case[str(row["case_id"])].append(row)
    macro: dict[str, Any] = {
        "model": "macro_over_five_models",
        "cases": len(case_ids),
        "model_case_rows": len(paired),
        "gold_fields_exposed_pct": 100 * mean(row["gold_fields_exposed"] for row in paired),
        "mean_candidate_field_count": mean(row["candidate_field_count"] for row in paired),
    }
    rng = random.Random(args.seed + 99)
    for metric in METRICS:
        query_value = mean(row[f"query_only_{metric}"] for row in paired)
        privileged_value = mean(row[f"privileged_{metric}"] for row in paired)
        boot = []
        for _ in range(args.bootstrap):
            sampled = rng.choices(case_ids, k=len(case_ids))
            values = [
                float(row[f"privileged_{metric}"]) - float(row[f"query_only_{metric}"])
                for case_id in sampled
                for row in by_case[case_id]
            ]
            boot.append(mean(values))
        macro[f"query_only_{metric}_pct"] = 100 * query_value
        macro[f"privileged_{metric}_pct"] = 100 * privileged_value
        macro[f"delta_{metric}_pp"] = 100 * (privileged_value - query_value)
        macro[f"delta_{metric}_ci_low_pp"] = 100 * percentile(boot, 0.025)
        macro[f"delta_{metric}_ci_high_pp"] = 100 * percentile(boot, 0.975)
    summaries.append(macro)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    fields = list(dict.fromkeys(key for row in summaries for key in row))
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    manifest = {
        "status": "complete",
        "benchmark": "VisEval",
        "scope": "independent-benchmark paired provenance diagnostic",
        "query_only_rows": len(base),
        "privileged_rows": len(privileged),
        "unique_cases": len(case_ids),
        "models": sorted(groups),
        "privileged_source": "candidate fields extracted from released SQL plus gold x/y metadata",
        "claim_boundary": "diagnoses target-derived field assistance; not a deployment score or a published-system reproduction",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
