#!/usr/bin/env python3
"""Analyze the strict-v3 nvBench-v1 transfer experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
if not BASE.exists():
    BASE = HERE.parent
sys.path.insert(0, str(BASE))

from common import best_graded_match, canonical_key, dedupe_specs  # noqa: E402
from metric_equivalence import core_key, difference_taxonomy  # noqa: E402


METRICS = ("exact_hit1", "exact_hit5", "core_hit1", "core_hit5", "top1_graded", "any_valid", "elapsed_seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * p
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def stratified_bootstrap(
    values_by_stratum: dict[str, list[float]], repetitions: int, seed: int
) -> tuple[float, float]:
    """Preserve the registered equal-family estimand in every resample."""
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        estimates.append(
            mean(
                mean(rng.choices(values, k=len(values)))
                for values in values_by_stratum.values()
            )
        )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def metrics(run: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    candidates = dedupe_specs(run.get("candidates", []))[:5]
    exact = [canonical_key(candidate) == canonical_key(gold) for candidate in candidates]
    core = [core_key(candidate) == core_key(gold) for candidate in candidates]
    return {
        "exact_hit1": float(bool(exact[:1]) and exact[0]),
        "exact_hit5": float(any(exact)),
        "core_hit1": float(bool(core[:1]) and core[0]),
        "core_hit5": float(any(core)),
        "top1_graded": best_graded_match(candidates[0], [gold])["macro"] if candidates else 0.0,
        "any_valid": float(bool(run.get("valid_candidates"))),
        "elapsed_seconds": float(run.get("elapsed_seconds", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-data", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--run-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    selected = json.loads(args.selected_data.read_text(encoding="utf-8"))
    selection_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    manual_audit = json.loads(args.manual_audit.read_text(encoding="utf-8"))
    gold = {str(row["record_id"]): dedupe_specs(row["gold_answer"])[0] for row in selected}
    metadata = {str(row["record_id"]): row for row in selected}

    runs: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    files = []
    for path in args.run_files:
        rows = read_jsonl(path)
        files.append({"file": path.name, "rows": len(rows)})
        for run in rows:
            runs[(str(run["model"]), str(run["condition"]))][str(run["record_id"])] = run

    per_case = []
    evaluated: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for key, by_id in sorted(runs.items()):
        model, condition = key
        if set(by_id) != set(gold):
            raise SystemExit(f"Incomplete {model}/{condition}: {len(by_id)} of {len(gold)} records")
        for record_id, run in sorted(by_id.items()):
            values = metrics(run, gold[record_id])
            evaluated[key][record_id] = values
            per_case.append({
                "record_id": record_id,
                "model": model,
                "condition": condition,
                "chart_family": metadata[record_id]["chart_family"],
                "hardness": metadata[record_id]["source_hardness"],
                **values,
            })

    summaries = []
    for (model, condition), cases in sorted(evaluated.items()):
        row: dict[str, Any] = {"model": model, "condition": condition, "n": len(cases)}
        for offset, metric in enumerate(METRICS):
            values_by_family: dict[str, list[float]] = defaultdict(list)
            for record_id, case in cases.items():
                values_by_family[str(metadata[record_id]["chart_family"])].append(case[metric])
            row[metric] = mean(mean(values) for values in values_by_family.values())
            low, high = stratified_bootstrap(values_by_family, args.bootstrap, args.seed + offset)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summaries.append(row)

    contrasts = []
    taxonomy = []
    for model in sorted({key[0] for key in evaluated}):
        available = {condition for candidate_model, condition in evaluated if candidate_model == model}
        for baseline, comparator, label in (
            ("direct", "direct_rich", "direct-rich minus direct-basic"),
            ("direct_rich", "staged", "staged-rich minus direct-rich"),
            ("direct", "staged", "staged-rich minus direct-basic (secondary)"),
        ):
            if not {baseline, comparator}.issubset(available):
                continue
            row: dict[str, Any] = {"model": model, "contrast": label, "n": len(gold)}
            for offset, metric in enumerate(METRICS):
                differences_by_family: dict[str, list[float]] = defaultdict(list)
                for record_id in gold:
                    differences_by_family[str(metadata[record_id]["chart_family"])].append(
                        evaluated[(model, comparator)][record_id][metric]
                        - evaluated[(model, baseline)][record_id][metric]
                    )
                row[f"delta_{metric}"] = mean(
                    mean(values) for values in differences_by_family.values()
                )
                low, high = stratified_bootstrap(
                    differences_by_family, args.bootstrap, args.seed + 100 + offset
                )
                row[f"delta_{metric}_ci_low"] = low
                row[f"delta_{metric}_ci_high"] = high
            contrasts.append(row)

            for record_id in sorted(gold):
                left = runs[(model, baseline)][record_id]
                right = runs[(model, comparator)][record_id]
                left_first = (left.get("candidates") or [None])[0]
                right_first = (right.get("candidates") or [None])[0]
                left_exact = canonical_key(left_first) == canonical_key(gold[record_id]) if left_first else False
                right_exact = canonical_key(right_first) == canonical_key(gold[record_id]) if right_first else False
                if right_exact and not left_exact:
                    taxonomy.append({
                        "record_id": record_id,
                        "model": model,
                        "contrast": label,
                        "chart_family": metadata[record_id]["chart_family"],
                        "hardness": metadata[record_id]["source_hardness"],
                        "baseline_difference": difference_taxonomy(left_first, gold[record_id]),
                        "baseline_core_equal_gold": core_key(left_first) == core_key(gold[record_id]) if left_first else False,
                        "baseline_top1": json.dumps(left_first, sort_keys=True),
                        "comparator_top1": json.dumps(right_first, sort_keys=True),
                        "gold": json.dumps(gold[record_id], sort_keys=True),
                    })

    taxonomy_summary = [
        {"model": model, "contrast": contrast, "baseline_difference": category, "cases": count}
        for (model, contrast, category), count in sorted(
            Counter((row["model"], row["contrast"], row["baseline_difference"]) for row in taxonomy).items()
        )
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case.csv", per_case)
    write_csv(args.output_dir / "condition_summary.csv", summaries)
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    write_csv(args.output_dir / "exact_gain_taxonomy.csv", taxonomy)
    write_csv(args.output_dir / "exact_gain_taxonomy_summary.csv", taxonomy_summary)
    write_csv(args.output_dir / "file_inventory.csv", files)
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps({
        "adapter_version": selection_manifest["adapter_version"],
        "selection_scope": selection_manifest["scope"],
        "source_records": selection_manifest["source_records"],
        "eligible_records": sum(selection_manifest["eligible_by_chart"].values()),
        "selected_records": selection_manifest["selection_count"],
        "selected_databases": selection_manifest["selection_databases"],
        "eligible_gold_validator_acceptance": "100% by construction; validator rejections are excluded and counted in the selection manifest",
        "manual_adapter_audit": {"sample_size": manual_audit["sample_size"], "passed": manual_audit["passed"], "failed": manual_audit["failed"]},
        "bootstrap_repetitions": args.bootstrap,
        "bootstrap_seed": args.seed,
        "bootstrap_unit": "case, stratified by chart family with equal family weighting",
        "files": files,
    }, indent=2), encoding="utf-8")
    print(f"Analyzed {len(per_case)} external case-run rows across {len(evaluated)} runs")


if __name__ == "__main__":
    main()
