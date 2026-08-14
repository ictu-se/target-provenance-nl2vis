from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
METRICS = ("any_valid_candidate", "raw_hit@1", "raw_hit@5", "raw_mrr", "top1_macro", "best5_macro", "elapsed_seconds")


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    location = (len(values) - 1) * p
    lower = int(location)
    upper = min(lower + 1, len(values) - 1)
    fraction = location - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def mcnemar(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def weighted_difference(pairs: list[dict[str, Any]], populations: dict[str, int], metric: str) -> float:
    strata: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        strata[pair["gold_family"]].append(pair[f"diff_{metric}"])
    return sum(populations[name] * mean(values) for name, values in strata.items()) / sum(populations.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=HERE / "design" / "forward_sample150.json")
    parser.add_argument("--output", type=Path, default=HERE / "out" / "a" / "paired_conditions.csv")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    with args.per_case.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    populations = {name: int(info["population"]) for name, info in design["strata"].items()}

    indexed = {
        (row["model"], row["seed"], row["temperature"], row["record_id"], row["condition"]): row
        for row in rows
    }
    run_keys = sorted({(row["model"], row["seed"], row["temperature"]) for row in rows})
    output: list[dict[str, Any]] = []
    rng = random.Random(20260807)
    for model, seed, temperature in run_keys:
        record_ids = sorted({row["record_id"] for row in rows if (row["model"], row["seed"], row["temperature"]) == (model, seed, temperature)})
        pairs: list[dict[str, Any]] = []
        for record_id in record_ids:
            direct = indexed.get((model, seed, temperature, record_id, "direct"))
            staged = indexed.get((model, seed, temperature, record_id, "staged"))
            if direct is None or staged is None:
                continue
            pair: dict[str, Any] = {"record_id": record_id, "gold_family": direct["gold_family"]}
            for metric in METRICS:
                pair[f"direct_{metric}"] = float(direct[metric])
                pair[f"staged_{metric}"] = float(staged[metric])
                pair[f"diff_{metric}"] = float(staged[metric]) - float(direct[metric])
            pairs.append(pair)
        if not pairs or set(pair["gold_family"] for pair in pairs) != set(populations):
            continue

        result: dict[str, Any] = {"model": model, "seed": seed, "temperature": temperature, "paired_n": len(pairs)}
        for metric in METRICS:
            result[f"direct_{metric}"] = weighted_difference(
                [{**pair, f"diff_{metric}": pair[f"direct_{metric}"]} for pair in pairs], populations, metric
            )
            result[f"staged_{metric}"] = weighted_difference(
                [{**pair, f"diff_{metric}": pair[f"staged_{metric}"]} for pair in pairs], populations, metric
            )
            result[f"staged_minus_direct_{metric}"] = weighted_difference(pairs, populations, metric)
            estimates: list[float] = []
            by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for pair in pairs:
                by_stratum[pair["gold_family"]].append(pair)
            for _ in range(args.bootstrap):
                sample = [item for group in by_stratum.values() for item in rng.choices(group, k=len(group))]
                estimates.append(weighted_difference(sample, populations, metric))
            result[f"staged_minus_direct_{metric}_ci_low"] = percentile(estimates, 0.025)
            result[f"staged_minus_direct_{metric}_ci_high"] = percentile(estimates, 0.975)
            if metric in {"any_valid_candidate", "raw_hit@1", "raw_hit@5"}:
                direct_only = sum(pair[f"direct_{metric}"] > pair[f"staged_{metric}"] for pair in pairs)
                staged_only = sum(pair[f"staged_{metric}"] > pair[f"direct_{metric}"] for pair in pairs)
                result[f"{metric}_direct_only"] = direct_only
                result[f"{metric}_staged_only"] = staged_only
                result[f"{metric}_mcnemar_p"] = mcnemar(direct_only, staged_only)
        result["staged_over_direct_latency_ratio"] = result["staged_elapsed_seconds"] / result["direct_elapsed_seconds"]
        output.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output:
        with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    else:
        args.output.write_text("", encoding="utf-8")
    print(f"Wrote {len(output)} paired condition comparisons")


if __name__ == "__main__":
    main()
