from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
METRICS = [
    "parse_success", "candidate_count", "valid_candidate_count", "any_valid_candidate",
    "all_candidates_valid", "elapsed_seconds", "raw_hit@1", "raw_hit@3", "raw_hit@5",
    "raw_recall@5", "raw_mrr", "valid_hit@1", "valid_hit@5", "valid_mrr",
    "top1_mark", "top1_channels", "top1_fields", "top1_operations", "top1_filters",
    "top1_macro", "best5_macro", "graded_ndcg@5",
]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def estimate(strata: dict[str, list[dict[str, str]]], populations: dict[str, int], metric: str) -> float:
    numerator = sum(populations[name] * mean(float(row[metric]) for row in cases) for name, cases in strata.items())
    denominator = sum(populations[name] for name in strata)
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case", type=Path, default=HERE / "outputs" / "forward_analysis" / "per_case.csv")
    parser.add_argument("--design", type=Path, default=HERE / "design" / "forward_sample150.json")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "forward_analysis" / "summary_design_weighted.csv")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    with args.per_case.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    populations = {name: int(info["population"]) for name, info in design["strata"].items()}
    expected = {name: int(info["sample"]) for name, info in design["strata"].items()}

    runs: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        runs[(row["model"], row["condition"], row["seed"], row["temperature"])].append(row)

    output: list[dict[str, Any]] = []
    rng = random.Random(20260807)
    for run_key, cases in sorted(runs.items()):
        strata: dict[str, list[dict[str, str]]] = defaultdict(list)
        for case in cases:
            strata[case["gold_family"]].append(case)
        if set(strata) != set(populations) or any(len(strata[name]) != expected[name] for name in populations):
            continue
        result: dict[str, Any] = dict(zip(("model", "condition", "seed", "temperature"), run_key))
        result.update({"n": len(cases), "weighted_population": sum(populations.values())})
        for metric in METRICS:
            result[metric] = estimate(strata, populations, metric)
            if metric in {"any_valid_candidate", "raw_hit@1", "raw_hit@5", "raw_mrr", "top1_macro", "best5_macro"}:
                estimates: list[float] = []
                for _ in range(args.bootstrap):
                    sampled = {
                        name: rng.choices(group, k=len(group))
                        for name, group in strata.items()
                    }
                    estimates.append(estimate(sampled, populations, metric))
                result[f"{metric}_ci_low"] = percentile(estimates, 0.025)
                result[f"{metric}_ci_high"] = percentile(estimates, 0.975)
        output.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output:
        with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    else:
        args.output.write_text("", encoding="utf-8")
    print(f"Wrote {len(output)} complete design-weighted runs")


if __name__ == "__main__":
    main()
