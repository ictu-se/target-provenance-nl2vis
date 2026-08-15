from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any

from common import best_graded_match, dedupe_specs, load_split, ranked_exact_metrics
from run_forward_ollama import DEFAULT_DATA


HERE = Path(__file__).resolve().parent
SUMMARY_METRICS = [
    "rerank_applicable", "parse_success", "raw_union_count", "invalid_unique_removed", "pool_size", "complete_pool_oracle",
    "rrf_hit@1", "rrf_hit@3", "rrf_hit@5", "rrf_mrr", "llm_hit@1", "llm_hit@3", "llm_hit@5",
    "llm_mrr", "pool_best_graded", "rrf_top1_graded", "rrf_best5_graded", "llm_top1_graded",
    "llm_best5_graded", "elapsed_seconds",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    location = (len(ordered) - 1) * p
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def graded_top1(predictions: list[dict[str, Any]], golds: list[dict[str, Any]]) -> float:
    return best_graded_match(predictions[0], golds)["macro"] if predictions else 0.0


def graded_best5(predictions: list[dict[str, Any]], golds: list[dict[str, Any]]) -> float:
    return max((best_graded_match(item, golds)["macro"] for item in predictions[:5]), default=0.0)


def case_metrics(run: dict[str, Any], golds: list[dict[str, Any]]) -> dict[str, Any]:
    pool = dedupe_specs(run.get("pool", []))
    rrf = dedupe_specs(run.get("rrf_top5", []))[:5]
    llm = dedupe_specs(run.get("llm_top5", []))[:5]
    pool_metrics = ranked_exact_metrics(pool, golds, range(1, 21))
    rrf_metrics = ranked_exact_metrics(rrf, golds, (1, 3, 5))
    llm_metrics = ranked_exact_metrics(llm, golds, (1, 3, 5))
    oracle = float(pool_metrics["MRR"] > 0.0)
    return {
        "record_id": run["record_id"], "index": run["index"], "model": run["model"],
        "pool_name": run.get("pool_name", "direct"),
        "seed": run["seed"], "temperature": run["temperature"],
        "gold_family": "+".join(sorted({str(gold.get("mark", "unknown")) for gold in golds})) or "unknown",
        "rerank_applicable": float(bool(pool)),
        "parse_success": float(bool(run.get("parse_success"))),
        "raw_union_count": int(run.get("raw_union_count", len(pool))),
        "invalid_unique_removed": int(run.get("invalid_unique_removed", 0)),
        "pool_size": len(pool),
        "complete_pool_oracle": oracle,
        **{f"pool_{key.lower()}": value for key, value in pool_metrics.items()},
        **{f"rrf_{key.lower()}": value for key, value in rrf_metrics.items()},
        **{f"llm_{key.lower()}": value for key, value in llm_metrics.items()},
        "pool_best_graded": max((best_graded_match(item, golds)["macro"] for item in pool), default=0.0),
        "rrf_top1_graded": graded_top1(rrf, golds), "rrf_best5_graded": graded_best5(rrf, golds),
        "llm_top1_graded": graded_top1(llm, golds), "llm_best5_graded": graded_best5(llm, golds),
        "elapsed_seconds": float(run.get("elapsed_seconds", 0.0)),
    }


def summarize(cases: list[dict[str, Any]], bootstrap: int) -> dict[str, Any]:
    output: dict[str, Any] = {
        "model": cases[0]["model"], "pool_name": cases[0]["pool_name"], "seed": cases[0]["seed"],
        "temperature": cases[0]["temperature"], "n": len(cases),
    }
    rng = random.Random(20260807)
    for metric in SUMMARY_METRICS:
        values = [float(case[metric]) for case in cases]
        output[metric] = mean(values)
        if metric in {"complete_pool_oracle", "rrf_hit@1", "rrf_hit@5", "llm_hit@1", "llm_hit@5", "llm_top1_graded"}:
            estimates = [mean(rng.choices(values, k=len(values))) for _ in range(bootstrap)]
            output[f"{metric}_ci_low"] = percentile(estimates, 0.025)
            output[f"{metric}_ci_high"] = percentile(estimates, 0.975)

    differences = [float(case["llm_hit@1"]) - float(case["rrf_hit@1"]) for case in cases]
    estimates = [mean(rng.choices(differences, k=len(differences))) for _ in range(bootstrap)]
    output["paired_llm_minus_rrf_hit1"] = mean(differences)
    output["paired_llm_minus_rrf_hit1_ci_low"] = percentile(estimates, 0.025)
    output["paired_llm_minus_rrf_hit1_ci_high"] = percentile(estimates, 0.975)
    output["llm_only_hit1"] = sum(case["llm_hit@1"] > case["rrf_hit@1"] for case in cases)
    output["rrf_only_hit1"] = sum(case["rrf_hit@1"] > case["llm_hit@1"] for case in cases)
    return output


def summarize_design(cases: list[dict[str, Any]], design: dict[str, Any], bootstrap: int) -> dict[str, Any] | None:
    if "population_strata_after_screen" in design:
        populations = {name: int(value) for name, value in design["population_strata_after_screen"].items()}
        expected = {name: int(value) for name, value in design["allocation"].items()}
    else:
        populations = {name: int(info["population"]) for name, info in design["strata"].items()}
        expected = {name: int(info["sample"]) for name, info in design["strata"].items()}
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        strata[case["gold_family"]].append(case)
    if set(strata) != set(populations) or any(len(strata[name]) != expected[name] for name in populations):
        return None

    def estimate(sampled: dict[str, list[dict[str, Any]]], metric: str) -> float:
        return sum(populations[name] * mean(float(case[metric]) for case in group) for name, group in sampled.items()) / sum(populations.values())

    def fpc_bootstrap(metric: str, rng: random.Random) -> list[float]:
        point = estimate(strata, metric)
        total = sum(populations.values())
        observed_means = {
            name: mean(float(case[metric]) for case in group) for name, group in strata.items()
        }
        values = []
        for _ in range(bootstrap):
            deviation = 0.0
            for name, population in populations.items():
                group = strata[name]
                n_h = len(group)
                if n_h <= 1 or population <= 1 or n_h >= population:
                    continue
                sampled_mean = mean(float(case[metric]) for case in rng.choices(group, k=n_h))
                scale = sqrt((1.0 - n_h / population) * n_h / (n_h - 1))
                deviation += (population / total) * scale * (sampled_mean - observed_means[name])
            values.append(point + deviation)
        return values

    output: dict[str, Any] = {
        "model": cases[0]["model"], "pool_name": cases[0]["pool_name"], "seed": cases[0]["seed"], "temperature": cases[0]["temperature"],
        "n": len(cases), "weighted_population": sum(populations.values()),
    }
    rng = random.Random(20260807)
    for metric in SUMMARY_METRICS:
        output[metric] = estimate(strata, metric)
        if metric in {"complete_pool_oracle", "rrf_hit@1", "rrf_hit@5", "llm_hit@1", "llm_hit@5", "llm_top1_graded"}:
            estimates = fpc_bootstrap(metric, rng)
            output[f"{metric}_ci_low"] = percentile(estimates, 0.025)
            output[f"{metric}_ci_high"] = percentile(estimates, 0.975)
    for case in cases:
        case["paired_llm_minus_rrf_hit1"] = float(case["llm_hit@1"]) - float(case["rrf_hit@1"])
    output["paired_llm_minus_rrf_hit1"] = estimate(strata, "paired_llm_minus_rrf_hit1")
    difference_estimates = fpc_bootstrap("paired_llm_minus_rrf_hit1", rng)
    output["paired_llm_minus_rrf_hit1_ci_low"] = percentile(difference_estimates, 0.025)
    output["paired_llm_minus_rrf_hit1_ci_high"] = percentile(difference_estimates, 0.975)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "out" / "r")
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "reranker_analysis")
    parser.add_argument("--design", type=Path, default=HERE / "design" / "forward_sample150.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    gold = {row["record_id"]: dedupe_specs(row["gold_answer"]) for row in load_split(args.data_dir, args.split)}
    cases: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.input_dir.glob(args.pattern)):
        for run in read_jsonl(path):
            if run["record_id"] not in gold:
                continue
            case = case_metrics(run, gold[run["record_id"]])
            cases.append(case)
            groups[(case["model"], case["pool_name"], int(case["seed"]), float(case["temperature"]))].append(case)
    summaries = [summarize(group, args.bootstrap) for group in groups.values()]
    design = json.loads(args.design.read_text(encoding="utf-8"))
    weighted_summaries = [summary for group in groups.values() if (summary := summarize_design(group, design, args.bootstrap)) is not None]
    curves: list[dict[str, Any]] = []
    for (model, pool_name, seed, temperature), group in groups.items():
        for k in range(1, 21):
            curves.append({
                "model": model, "pool_name": pool_name, "seed": seed, "temperature": temperature, "K": k, "n": len(group),
                "pool_hit": mean(float(case[f"pool_hit@{k}"]) for case in group),
                "pool_recall": mean(float(case[f"pool_recall@{k}"]) for case in group),
                "pool_precision": mean(float(case[f"pool_precision@{k}"]) for case in group),
                "pool_f1": mean(float(case[f"pool_f1@{k}"]) for case in group),
            })
    write_csv(args.output_dir / "per_case.csv", cases)
    write_csv(args.output_dir / "summary.csv", summaries)
    write_csv(args.output_dir / "summary_design_weighted.csv", weighted_summaries)
    write_csv(args.output_dir / "coverage_curve.csv", curves)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"files": sorted(path.name for path in args.input_dir.glob(args.pattern)), "bootstrap": args.bootstrap,
                    "bootstrap_seed": 20260807, "n_case_runs": len(cases), "split": args.split,
                    "design": str(args.design),
                    "estimator": "stratified Horvitz-Thompson population mean, equivalent to the known-N post-stratified/Hajek mean",
                    "uncertainty": "within-stratum fixed-size empirical bootstrap with finite-population-adjusted replicate deviations"}, indent=2), encoding="utf-8"
    )
    print(f"Analyzed {len(cases)} reranker case-runs")


if __name__ == "__main__":
    main()
