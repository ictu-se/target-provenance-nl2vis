from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from common import canonical_key, dedupe_specs, load_split, ranked_exact_metrics
from run_forward_ollama import DEFAULT_DATA


HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keys(specs: list[dict[str, Any]]) -> list[str]:
    return [key for spec in dedupe_specs(specs) if (key := canonical_key(spec)) is not None]


def jaccard(left: list[str], right: list[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def rbo(left: list[str], right: list[str], p: float = 0.9) -> float:
    depth = max(len(left), len(right))
    if depth == 0:
        return 1.0
    score = 0.0
    left_prefix: set[str] = set()
    right_prefix: set[str] = set()
    for d in range(1, depth + 1):
        if d <= len(left):
            left_prefix.add(left[d - 1])
        if d <= len(right):
            right_prefix.add(right[d - 1])
        score += (1 - p) * (p ** (d - 1)) * len(left_prefix & right_prefix) / d
    score += (p**depth) * len(left_prefix & right_prefix) / depth
    return score


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "out" / "f")
    parser.add_argument("--pattern", default="qwen3*_repeat150.jsonl")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "stability")
    args = parser.parse_args()

    gold = {row["record_id"]: dedupe_specs(row["gold_answer"]) for row in load_split(args.data_dir, "test")}
    rows = [row for path in sorted(args.input_dir.glob(args.pattern)) for row in read_jsonl(path)]
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["condition"], row["record_id"])][int(row["seed"])] = row

    per_pair: list[dict[str, Any]] = []
    run_metrics: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for (condition, record_id), by_seed in grouped.items():
        for seed, row in by_seed.items():
            metrics = ranked_exact_metrics(dedupe_specs(row.get("candidates", [])), gold[record_id], (1, 5))
            run_metrics[(condition, seed)].append(metrics)
        for left_seed, right_seed in combinations(sorted(by_seed), 2):
            left = keys(by_seed[left_seed].get("candidates", []))
            right = keys(by_seed[right_seed].get("candidates", []))
            left_valid = keys(by_seed[left_seed].get("valid_candidates", []))
            right_valid = keys(by_seed[right_seed].get("valid_candidates", []))
            per_pair.append({
                "condition": condition, "record_id": record_id,
                "left_seed": left_seed, "right_seed": right_seed,
                "top1_agreement": float(bool(left) and bool(right) and left[0] == right[0]),
                "candidate_set_jaccard": jaccard(left, right),
                "valid_set_jaccard": jaccard(left_valid, right_valid),
                "rank_biased_overlap_p90": rbo(left, right, 0.9),
            })

    summary: list[dict[str, Any]] = []
    for condition in sorted({row["condition"] for row in per_pair}):
        pairs = [row for row in per_pair if row["condition"] == condition]
        result: dict[str, Any] = {"condition": condition, "pair_case_rows": len(pairs)}
        for metric in ("top1_agreement", "candidate_set_jaccard", "valid_set_jaccard", "rank_biased_overlap_p90"):
            result[metric] = mean(float(row[metric]) for row in pairs)
        seeds = sorted(seed for candidate_condition, seed in run_metrics if candidate_condition == condition)
        result["seeds"] = ";".join(str(seed) for seed in seeds)
        for metric in ("Hit@1", "Hit@5", "MRR"):
            values = [mean(row[metric] for row in run_metrics[(condition, seed)]) for seed in seeds]
            result[f"{metric.lower()}_across_seed_mean"] = mean(values)
            result[f"{metric.lower()}_across_seed_sd"] = pstdev(values) if len(values) > 1 else 0.0
            result[f"{metric.lower()}_across_seed_min"] = min(values)
            result[f"{metric.lower()}_across_seed_max"] = max(values)
        summary.append(result)

    write_csv(args.output_dir / "per_seed_pair_case.csv", per_pair)
    write_csv(args.output_dir / "summary.csv", summary)
    print(f"Analyzed {len(rows)} run rows and {len(per_pair)} seed-pair cases")


if __name__ == "__main__":
    main()
