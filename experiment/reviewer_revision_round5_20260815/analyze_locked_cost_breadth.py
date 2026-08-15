#!/usr/bin/env python3
"""Surface locked core equivalence, cost, and matched breadth measures."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
ROUND2 = HERE.parent / "reviewer_revision_round2_20260814"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROUND2))

from common import canonical_key, dedupe_specs, load_split, spec_components  # noqa: E402
from metric_equivalence import core_key  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_f1(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision, recall = overlap / len(left), overlap / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def component_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = spec_components(left), spec_components(right)
    return mean(set_f1(a[name], b[name]) for name in ("mark", "channels", "fields", "operations", "filters"))


def maximum_matching(predictions: list[dict[str, Any]], golds: list[dict[str, Any]], predicate: Callable[[dict[str, Any], dict[str, Any]], bool]) -> int:
    adjacency = [[index for index, gold in enumerate(golds) if predicate(prediction, gold)] for prediction in predictions]
    matched_gold: dict[int, int] = {}

    def augment(prediction_index: int, seen: set[int]) -> bool:
        for gold_index in adjacency[prediction_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            if gold_index not in matched_gold or augment(matched_gold[gold_index], seen):
                matched_gold[gold_index] = prediction_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(predictions)))


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def weighted(cases: list[dict[str, Any]], populations: dict[str, int], metric: str) -> float:
    strata: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        strata[str(case["gold_family"])].append(float(case[metric]))
    total = sum(populations.values())
    return sum(populations[name] * mean(strata[name]) for name in populations) / total


def interval(cases: list[dict[str, Any]], populations: dict[str, int], metric: str, repetitions: int, seed: int) -> tuple[float, float]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        strata[str(case["gold_family"])].append(case)
    point = weighted(cases, populations, metric)
    means = {name: mean(float(row[metric]) for row in group) for name, group in strata.items()}
    total = sum(populations.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        deviation = 0.0
        for name, population in populations.items():
            group = strata[name]
            n_h = len(group)
            if n_h <= 1 or n_h >= population:
                continue
            sampled = mean(float(row[metric]) for row in rng.choices(group, k=n_h))
            scale = sqrt((1 - n_h / population) * n_h / (n_h - 1))
            deviation += population / total * scale * (sampled - means[name])
        estimates.append(point + deviation)
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    design = json.loads(args.design.read_text(encoding="utf-8"))
    populations = {str(key): int(value) for key, value in design["population_strata_after_screen"].items()}
    indices = {int(value) for value in design["indices"]}
    gold_rows = {row["record_id"]: row for row in load_split(args.data_dir, "dev") if int(row["index"]) in indices}
    cases = []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        for run in read_jsonl(path):
            gold_row = gold_rows[str(run["record_id"])]
            golds = dedupe_specs(gold_row["gold_answer"])
            predictions = dedupe_specs(run.get("candidates", []))[:5]
            gold_family = "+".join(sorted({str(gold.get("mark", "unknown")) for gold in golds}))
            exact = maximum_matching(predictions, golds, lambda a, b: canonical_key(a) == canonical_key(b))
            core = maximum_matching(predictions, golds, lambda a, b: core_key(a) == core_key(b))
            components = maximum_matching(predictions, golds, lambda a, b: component_similarity(a, b) == 1.0)
            matched_80 = maximum_matching(predictions, golds, lambda a, b: component_similarity(a, b) >= 0.80)
            matched_90 = maximum_matching(predictions, golds, lambda a, b: component_similarity(a, b) >= 0.90)
            metadata = run.get("ollama_metadata") or {}
            denominator = max(1, len(golds))
            cases.append({
                "record_id": run["record_id"], "model": run["model"], "condition": run["condition"],
                "gold_family": gold_family, "gold_count": len(golds), "candidate_count": len(predictions),
                "exact_recall@5": exact / denominator, "core_recall@5": core / denominator,
                "component_identical_recall@5": components / denominator,
                "matched_recall@5_tau80": matched_80 / denominator,
                "matched_recall@5_tau90": matched_90 / denominator,
                "exact_recovered": exact, "core_recovered": core, "component_identical_recovered": components,
                "matched_recovered_tau80": matched_80, "matched_recovered_tau90": matched_90,
                "elapsed_seconds": float(run.get("elapsed_seconds") or 0.0),
                "prompt_tokens": int(metadata.get("prompt_eval_count") or 0),
                "generated_tokens": int(metadata.get("eval_count") or 0),
                "total_tokens": int(metadata.get("prompt_eval_count") or 0) + int(metadata.get("eval_count") or 0),
            })

    metrics = [
        "exact_recall@5", "core_recall@5", "component_identical_recall@5",
        "matched_recall@5_tau80", "matched_recall@5_tau90", "exact_recovered",
        "core_recovered", "component_identical_recovered", "matched_recovered_tau80",
        "matched_recovered_tau90", "elapsed_seconds", "prompt_tokens", "generated_tokens", "total_tokens",
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[(str(case["model"]), str(case["condition"]))].append(case)
    summary = []
    for (model, condition), group in sorted(groups.items()):
        if len(group) != 150:
            raise SystemExit(f"{model}/{condition}: expected 150 cases, found {len(group)}")
        row: dict[str, Any] = {"model": model, "condition": condition, "n": len(group)}
        for offset, metric in enumerate(metrics):
            row[metric] = weighted(group, populations, metric)
            low, high = interval(group, populations, metric, args.bootstrap, 20260815 + offset)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summary.append(row)

    write_csv(args.output_dir / "per_case.csv", cases)
    write_csv(args.output_dir / "condition_summary.csv", summary)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "rows": len(cases), "runs": len(groups), "bootstrap": args.bootstrap,
        "core_policy": "presentation metadata removed exactly as registered before analysis",
        "component_identical_policy": "maximum bipartite matching where all five registered component-set similarities equal one",
        "threshold_policy": "maximum bipartite matching at equal-weight component macro thresholds 0.80 and 0.90",
        "cost_policy": "retained wall-clock elapsed_seconds and Ollama prompt/evaluation token counts",
        "uncertainty": "percentile endpoints of the finite-population-adjusted within-stratum empirical bootstrap",
    }, indent=2), encoding="utf-8")
    print(f"Analyzed {len(cases)} locked case-runs")


if __name__ == "__main__":
    main()

