#!/usr/bin/env python3
"""Analyze eligible full and leave-self-out reranking runs."""

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
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
sys.path.insert(0, str(BASE))

from common import best_graded_match, canonical_key, dedupe_specs, load_split, ranked_exact_metrics  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values); position = (len(ordered) - 1) * p
    lower = int(position); upper = min(lower + 1, len(ordered) - 1); fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def ancestry(run: dict[str, Any], spec: dict[str, Any] | None) -> set[str]:
    if spec is None:
        return set()
    key = canonical_key(spec)
    for detail in run.get("pool_details", []):
        if detail.get("canonical_key") == key:
            return {str(source["model"]) for source in detail.get("sources", [])}
    return set()


def case_metrics(run: dict[str, Any], golds: list[dict[str, Any]]) -> dict[str, Any]:
    pool = dedupe_specs(run.get("pool", []))
    rrf = dedupe_specs(run.get("rrf_top5", []))[:5]
    llm = dedupe_specs(run.get("llm_top5", []))[:5]
    pool_metrics = ranked_exact_metrics(pool, golds, range(1, max(2, len(pool) + 1)))
    rrf_metrics = ranked_exact_metrics(rrf, golds, (1, 5))
    llm_metrics = ranked_exact_metrics(llm, golds, (1, 5))
    sources = ancestry(run, llm[0] if llm else None)
    return {
        "record_id": run["record_id"], "model": run["model"], "variant": run["variant"],
        "gold_family": "+".join(sorted({str(gold.get("mark", "unknown")) for gold in golds})),
        "pool_size": len(pool), "pool_oracle": float(pool_metrics["MRR"] > 0),
        "rrf_hit@1": rrf_metrics["Hit@1"], "rrf_hit@5": rrf_metrics["Hit@5"],
        "llm_hit@1": llm_metrics["Hit@1"], "llm_hit@5": llm_metrics["Hit@5"],
        "rrf_top1_graded": best_graded_match(rrf[0], golds)["macro"] if rrf else 0.0,
        "llm_top1_graded": best_graded_match(llm[0], golds)["macro"] if llm else 0.0,
        "parse_success": float(bool(run.get("parse_success"))),
        "prefix_complete": float(bool(run.get("prefix_complete"))),
        "returned_count": len(llm), "target_count": min(5, len(pool)),
        "self_source_any": float(str(run["model"]) in sources),
        "self_source_exclusive": float(sources == {str(run["model"])}),
        "source_gemma": float("gemma3:27b" in sources), "source_llama": float("llama3.2:3b" in sources),
        "source_mistral": float("mistral-small:24b" in sources), "source_qwen": float("qwen3:14b" in sources),
        "elapsed_seconds": float(run.get("elapsed_seconds") or 0.0),
    }


def summarize(group: list[dict[str, Any]], populations: dict[str, int], bootstrap: int) -> dict[str, Any]:
    for case in group:
        case["llm_minus_rrf_hit1"] = case["llm_hit@1"] - case["rrf_hit@1"]
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in group: strata[case["gold_family"]].append(case)
    total = sum(populations.values())
    def estimate(metric: str) -> float:
        return sum(populations[name] * mean(float(case[metric]) for case in strata[name]) for name in populations) / total
    metrics = [
        "pool_size", "pool_oracle", "rrf_hit@1", "rrf_hit@5", "llm_hit@1", "llm_hit@5",
        "rrf_top1_graded", "llm_top1_graded", "parse_success", "prefix_complete", "returned_count",
        "target_count", "self_source_any", "self_source_exclusive", "source_gemma", "source_llama",
        "source_mistral", "source_qwen", "elapsed_seconds", "llm_minus_rrf_hit1",
    ]
    output: dict[str, Any] = {"model": group[0]["model"], "variant": group[0]["variant"], "n": len(group)}
    rng = random.Random(20260815)
    for metric in metrics:
        point = estimate(metric); output[metric] = point
        observed = {name: mean(float(case[metric]) for case in rows) for name, rows in strata.items()}
        estimates = []
        for _ in range(bootstrap):
            deviation = 0.0
            for name, population in populations.items():
                rows = strata[name]; n_h = len(rows)
                if n_h <= 1 or n_h >= population: continue
                sampled = mean(float(case[metric]) for case in rng.choices(rows, k=n_h))
                deviation += population / total * sqrt((1 - n_h / population) * n_h / (n_h - 1)) * (sampled - observed[name])
            estimates.append(point + deviation)
        output[f"{metric}_ci_low"] = percentile(estimates, 0.025)
        output[f"{metric}_ci_high"] = percentile(estimates, 0.975)
    nonempty = output["parse_success"]
    for metric in ("self_source_any", "self_source_exclusive", "source_gemma", "source_llama", "source_mistral", "source_qwen"):
        output[f"{metric}_among_nonempty"] = output[metric] / nonempty if nonempty else 0.0
    return output


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
    gold = {row["record_id"]: dedupe_specs(row["gold_answer"]) for row in load_split(args.data_dir, "dev")}
    cases = []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        for run in read_jsonl(path):
            cases.append(case_metrics(run, gold[run["record_id"]]))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases: groups[(str(case["model"]), str(case["variant"]))].append(case)
    summaries = []
    for key, group in sorted(groups.items()):
        if len(group) != 150: raise SystemExit(f"{key}: expected 150, found {len(group)}")
        summaries.append(summarize(group, populations, args.bootstrap))
    write_csv(args.output_dir / "per_case.csv", cases)
    write_csv(args.output_dir / "summary.csv", summaries)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "case_runs": len(cases), "groups": len(groups), "bootstrap": args.bootstrap,
        "uncertainty": "percentile endpoints of finite-population-adjusted stratified empirical bootstrap",
        "ancestry": "a selected candidate may have multiple generator sources; source-any indicators are nonexclusive",
    }, indent=2), encoding="utf-8")
    print(f"Analyzed {len(cases)} reranker case-runs")


if __name__ == "__main__":
    main()
