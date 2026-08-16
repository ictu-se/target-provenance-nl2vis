#!/usr/bin/env python3
"""Condition ordering on oracle coverage and surface generation breadth."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
if not (BASE / "common.py").exists():
    BASE = HERE.parent
sys.path.insert(0, str(BASE))

from common import canonical_key, dedupe_specs, load_split  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def family(golds: list[dict[str, Any]]) -> str:
    return "+".join(sorted({str(item.get("mark", "unknown")) for item in golds}))


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")))


AGGREGATE_CUES = {
    "count": {"count", "number", "many", "frequency", "frequencies"},
    "mean": {"average", "mean"},
    "sum": {"sum", "total"},
    "min": {"minimum", "lowest", "min"},
    "max": {"maximum", "highest", "max"},
    "median": {"median"},
}
MARK_CUES = {
    "bar": {"bar", "column"},
    "line": {"line", "trend"},
    "arc": {"pie", "donut", "arc"},
    "point": {"scatter", "point"},
    "rect": {"heatmap", "matrix"},
    "boxplot": {"boxplot", "box", "distribution"},
}


def spec_only_score(query: str, spec: dict[str, Any]) -> float:
    """Pre-gold lexical/structural score using no source-rank information."""
    query_tokens = token_set(query)
    score = 0.0
    encoding = spec.get("encoding") if isinstance(spec.get("encoding"), dict) else {}
    for definition in encoding.values():
        if not isinstance(definition, dict):
            continue
        field = definition.get("field")
        if isinstance(field, str):
            field_tokens = token_set(field)
            score += 2.0 * float(bool(field_tokens) and field_tokens <= query_tokens)
            score += 0.25 * len(field_tokens & query_tokens)
        aggregate = definition.get("aggregate")
        if isinstance(aggregate, str) and query_tokens & AGGREGATE_CUES.get(aggregate.lower(), set()):
            score += 1.0
        time_unit = definition.get("timeUnit")
        if isinstance(time_unit, str) and time_unit.lower() in query_tokens:
            score += 0.75
        if definition.get("bin") and query_tokens & {"bin", "histogram", "distribution", "range"}:
            score += 0.75
    mark = spec.get("mark")
    if isinstance(mark, dict):
        mark = mark.get("type")
    if isinstance(mark, str) and query_tokens & MARK_CUES.get(mark.lower(), set()):
        score += 1.0
    transform = spec.get("transform")
    if isinstance(transform, list) and query_tokens & {"where", "only", "filter", "excluding", "include"}:
        score += 0.5
    return score


def heuristic_rank(query: str, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [(spec_only_score(query, spec), canonical_key(spec) or "", spec) for spec in pool]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [spec for _, _, spec in scored]


def ranking_metrics(ranked: list[dict[str, Any]], pool: list[dict[str, Any]], golds: list[dict[str, Any]]) -> dict[str, float]:
    gold_keys = {canonical_key(item) for item in golds}
    gold_keys.discard(None)
    pool_relevant = sum((canonical_key(item) in gold_keys) for item in dedupe_specs(pool))
    relevance = [float(canonical_key(item) in gold_keys) for item in dedupe_specs(ranked)[:5]]
    hit1 = relevance[0] if relevance else 0.0
    hit5 = float(any(relevance))
    first = next((index for index, value in enumerate(relevance, start=1) if value), None)
    mrr = 1.0 / first if first else 0.0
    dcg = sum(value / math.log2(index + 1) for index, value in enumerate(relevance, start=1))
    ideal_count = min(5, pool_relevant)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_count + 1))
    return {
        "hit1": hit1,
        "hit5": hit5,
        "mrr": mrr,
        "ndcg5": dcg / idcg if idcg else 0.0,
        "oracle": float(pool_relevant > 0),
        "pool_relevant_count": float(pool_relevant),
    }


def case_weights(rows: list[dict[str, Any]], populations: dict[str, int]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["gold_family"])] += 1
    return {name: populations[name] / counts[name] for name in counts}


def weighted(rows: list[dict[str, Any]], metric: str, populations: dict[str, int], conditioned: bool = False) -> float:
    weights = case_weights(rows, populations)
    selected = [row for row in rows if not conditioned or float(row["oracle"]) == 1.0]
    denominator = sum(weights[str(row["gold_family"])] for row in selected)
    return sum(weights[str(row["gold_family"])] * float(row[metric]) for row in selected) / denominator if denominator else 0.0


def analyze_ordering(rerank_dir: Path, data_dir: Path, design: dict[str, Any], output_dir: Path) -> None:
    populations = {str(key): int(value) for key, value in design["population_strata_after_screen"].items()}
    gold_rows = {row["record_id"]: row for row in load_split(data_dir, "dev")}
    cases: list[dict[str, Any]] = []
    for path in sorted(rerank_dir.glob("*_91_all_taf_eligible_full.jsonl")):
        for run in read_jsonl(path):
            row = gold_rows[str(run["record_id"])]
            golds = dedupe_specs(row["gold_answer"])
            pool = dedupe_specs(run.get("pool", []))
            rankings = {
                "rrf_ensemble": dedupe_specs(run.get("rrf_top5", [])),
                "spec_only_llm": dedupe_specs(run.get("llm_top5", [])),
                "spec_only_heuristic": heuristic_rank(str(row.get("nl_query", "")), pool)[:5],
            }
            for ranker, ranking in rankings.items():
                metrics = ranking_metrics(ranking, pool, golds)
                cases.append({
                    "record_id": run["record_id"],
                    "model": run["model"],
                    "ranker": ranker,
                    "gold_family": family(golds),
                    "pool_size": len(pool),
                    **metrics,
                })
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[(str(case["model"]), str(case["ranker"]))].append(case)
    summary = []
    for (model, ranker), rows in sorted(groups.items()):
        if len(rows) != 150:
            raise SystemExit(f"{model}/{ranker}: expected 150 rows, found {len(rows)}")
        oracle_raw_n = sum(int(row["oracle"]) for row in rows)
        result: dict[str, Any] = {
            "model": model,
            "ranker": ranker,
            "n_all": len(rows),
            "n_oracle_positive": oracle_raw_n,
            "design_weighted_oracle_coverage": weighted(rows, "oracle", populations),
        }
        for metric in ("hit1", "hit5", "mrr", "ndcg5"):
            result[f"all_{metric}"] = weighted(rows, metric, populations)
            result[f"oracle_conditioned_{metric}"] = weighted(rows, metric, populations, conditioned=True)
        summary.append(result)
    write_csv(output_dir / "ordering_per_case.csv", cases)
    write_csv(output_dir / "ordering_summary.csv", summary)


def analyze_breadth(generation_dir: Path, data_dir: Path, design: dict[str, Any], output_dir: Path) -> None:
    populations = {str(key): int(value) for key, value in design["population_strata_after_screen"].items()}
    indices = {int(value) for value in design["indices"]}
    index_family = {
        int(row["index"]): family(dedupe_specs(row["gold_answer"]))
        for row in load_split(data_dir, "dev")
        if int(row["index"]) in indices
    }
    cases = []
    for path in sorted(generation_dir.glob("*.jsonl")):
        for run in read_jsonl(path):
            index = int(run["index"])
            if index not in indices:
                continue
            candidates = list(run.get("candidates", []))
            unique = dedupe_specs(candidates)
            metadata = run.get("ollama_metadata") or {}
            done_reason = str(metadata.get("done_reason") or "missing")
            eval_count = int(metadata.get("eval_count") or 0)
            cases.append({
                "record_id": run["record_id"],
                "model": run["model"],
                "condition": run["condition"],
                "gold_family": index_family[index],
                "emitted_count": len(candidates),
                "unique_count": len(unique),
                "done_reason": done_reason,
                "generated_tokens": eval_count,
                "token_limit_completion": float(done_reason in {"length", "max_tokens"} or eval_count >= 3072),
            })
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[(str(case["model"]), str(case["condition"]))].append(case)
    summary = []
    for (model, condition), rows in sorted(groups.items()):
        if len(rows) != 150:
            raise SystemExit(f"{model}/{condition}: expected 150 rows, found {len(rows)}")
        result: dict[str, Any] = {
            "model": model,
            "condition": condition,
            "n": len(rows),
            "mean_emitted_count": weighted(rows, "emitted_count", populations),
            "mean_unique_count": weighted(rows, "unique_count", populations),
            "max_generated_tokens": max(int(row["generated_tokens"]) for row in rows),
            "token_limit_completions": sum(int(row["token_limit_completion"]) for row in rows),
            "done_reason_stop": sum(row["done_reason"] == "stop" for row in rows),
        }
        for count in range(0, 6):
            for row in rows:
                row[f"emit_{count}"] = float(int(row["emitted_count"]) == count)
            result[f"proportion_emit_{count}"] = weighted(rows, f"emit_{count}", populations)
        summary.append(result)
    write_csv(output_dir / "breadth_per_case.csv", cases)
    write_csv(output_dir / "breadth_summary.csv", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank-dir", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyze_ordering(args.rerank_dir, args.data_dir, design, args.output_dir)
    analyze_breadth(args.generation_dir, args.data_dir, design, args.output_dir)
    manifest = {
        "status": "complete",
        "ordering": "all-case and oracle-conditioned exact relevance metrics for fixed eligible pools",
        "ndcg": "binary exact-match relevance; IDCG uses the number of exact gold candidates present in each pool, capped at five",
        "spec_only_heuristic": "query/candidate lexical-structural score with canonical-key tie breaking; no source identity, source rank, RRF score, or gold input",
        "breadth": "design-weighted emitted and unique candidate counts plus retained Ollama completion metadata",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
