from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from common import (
    best_graded_match,
    canonical_key,
    dedupe_specs,
    load_split,
    ranked_exact_metrics,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
DEFAULT_DATA = Path(os.environ.get(
    "NVBENCH2_DATA_DIR",
    WORKSPACE / "data_benchmarks" / "datasets" / "nvBench-2.0" / "data" / "nvbench2.0",
))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def gold_family(golds: list[dict[str, Any]]) -> str:
    marks = sorted({str(gold.get("mark", "unknown")) for gold in golds})
    return "+".join(marks) if marks else "unknown"


def evaluate_case(run_row: dict[str, Any], gold_row: dict[str, Any]) -> dict[str, Any]:
    golds = dedupe_specs(gold_row["gold_answer"])
    raw = dedupe_specs(run_row.get("candidates", []))[:5]
    valid = dedupe_specs(run_row.get("valid_candidates", []))[:5]
    raw_exact = ranked_exact_metrics(raw, golds, (1, 3, 5))
    valid_exact = ranked_exact_metrics(valid, golds, (1, 3, 5))

    top1 = best_graded_match(raw[0], golds) if raw else {
        name: 0.0 for name in ("mark", "channels", "fields", "operations", "filters", "macro")
    }
    best5 = max(
        (best_graded_match(candidate, golds) for candidate in raw[:5]),
        key=lambda score: score["macro"],
        default={name: 0.0 for name in ("mark", "channels", "fields", "operations", "filters", "macro")},
    )
    exact_keys = {canonical_key(gold) for gold in golds}
    relevance = [best_graded_match(candidate, golds)["macro"] for candidate in raw[:5]]
    dcg = sum((2**gain - 1) / __import__("math").log2(rank + 2) for rank, gain in enumerate(relevance))
    ideal_gains = [1.0] * min(5, len(exact_keys)) + [0.0] * max(0, 5 - len(exact_keys))
    idcg = sum((2**gain - 1) / __import__("math").log2(rank + 2) for rank, gain in enumerate(ideal_gains))

    return {
        "record_id": run_row["record_id"],
        "index": run_row["index"],
        "model": run_row["model"],
        "condition": run_row["condition"],
        "seed": run_row["seed"],
        "temperature": run_row.get("temperature", 0.0),
        "parse_success": float(bool(run_row.get("parse_success"))),
        "candidate_count": len(raw),
        "valid_candidate_count": len(valid),
        "any_candidate": float(bool(raw)),
        "any_valid_candidate": float(bool(valid)),
        "all_candidates_valid": float(bool(raw) and len(raw) == len(valid)),
        "elapsed_seconds": float(run_row.get("elapsed_seconds", 0.0)),
        "gold_count_raw": len(gold_row["gold_answer"]),
        "gold_count_canonical": len(golds),
        "ambiguity_group": "multi_gold" if len(golds) > 1 else "single_gold",
        "gold_family": gold_family(golds),
        **{f"raw_{key.lower()}": value for key, value in raw_exact.items()},
        **{f"valid_{key.lower()}": value for key, value in valid_exact.items()},
        **{f"top1_{key}": value for key, value in top1.items()},
        **{f"best5_{key}": value for key, value in best5.items()},
        "graded_ndcg@5": dcg / idcg if idcg else 0.0,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(values: list[float], rng: random.Random, repetitions: int) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    estimates = [mean(rng.choices(values, k=len(values))) for _ in range(repetitions)]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    columns = list(materialized[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def summarize(groups: dict[tuple[Any, ...], list[dict[str, Any]]], group_names: list[str], repetitions: int) -> list[dict[str, Any]]:
    metrics = [
        "parse_success", "candidate_count", "valid_candidate_count", "any_valid_candidate",
        "all_candidates_valid", "elapsed_seconds", "raw_hit@1", "raw_hit@3", "raw_hit@5",
        "raw_recall@5", "raw_mrr", "valid_hit@1", "valid_hit@5", "valid_mrr",
        "top1_mark", "top1_channels", "top1_fields", "top1_operations", "top1_filters",
        "top1_macro", "best5_macro", "graded_ndcg@5",
    ]
    output: list[dict[str, Any]] = []
    for group_key, cases in sorted(groups.items(), key=lambda pair: tuple(str(value) for value in pair[0])):
        row: dict[str, Any] = dict(zip(group_names, group_key))
        row["n"] = len(cases)
        rng = random.Random(20260807)
        for metric in metrics:
            values = [float(case[metric]) for case in cases]
            row[metric] = mean(values)
            if metric in {"raw_hit@1", "raw_hit@5", "raw_mrr", "top1_macro", "best5_macro", "any_valid_candidate"}:
                low, high = bootstrap_mean_ci(values, rng, repetitions)
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "out" / "f")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "forward_analysis")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--file-pattern", default="*.jsonl")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    gold_rows = {row["record_id"]: row for row in load_split(args.data_dir, args.split)}
    per_case: list[dict[str, Any]] = []
    file_inventory: list[dict[str, Any]] = []
    for path in sorted(args.input_dir.glob(args.file_pattern)):
        run_rows = read_jsonl(path)
        matched = 0
        for run_row in run_rows:
            gold_row = gold_rows.get(run_row.get("record_id"))
            if gold_row is None:
                continue
            per_case.append(evaluate_case(run_row, gold_row))
            matched += 1
        file_inventory.append({"file": path.name, "rows": len(run_rows), "matched_gold_rows": matched})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case.csv", per_case)
    write_csv(args.output_dir / "file_inventory.csv", file_inventory)

    run_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    subgroup_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    family_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for case in per_case:
        run_key = (case["model"], case["condition"], case["seed"], case["temperature"])
        run_groups[run_key].append(case)
        subgroup_groups[run_key + (case["ambiguity_group"],)].append(case)
        family_groups[run_key + (case["gold_family"],)].append(case)

    summary = summarize(run_groups, ["model", "condition", "seed", "temperature"], args.bootstrap)
    subgroup = summarize(
        subgroup_groups,
        ["model", "condition", "seed", "temperature", "ambiguity_group"],
        args.bootstrap,
    )
    family = summarize(
        family_groups,
        ["model", "condition", "seed", "temperature", "gold_family"],
        args.bootstrap,
    )
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "by_ambiguity.csv", subgroup)
    write_csv(args.output_dir / "by_chart_family.csv", family)
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "input_dir": str(args.input_dir.resolve()),
                "split": args.split,
                "files": file_inventory,
                "case_rows": len(per_case),
                "bootstrap_repetitions": args.bootstrap,
                "bootstrap_seed": 20260807,
                "exact_match_policy": "canonicalized full-spec equality against all unique gold specifications",
                "graded_policy": "unweighted macro F1 over mark, channels, channel-field pairs, operations, and filters",
                "validity_policy": "allowed mark, required channels, schema-grounded fields, and supported filters",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Analyzed {len(per_case)} case-run rows across {len(run_groups)} runs")


if __name__ == "__main__":
    main()
