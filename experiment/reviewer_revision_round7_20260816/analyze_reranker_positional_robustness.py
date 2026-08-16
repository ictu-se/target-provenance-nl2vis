#!/usr/bin/env python3
"""Analyze reranker stability across locked candidate presentation orders."""

from __future__ import annotations

import argparse
import csv
import json
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

from common import canonical_key, dedupe_specs, load_split, ranked_exact_metrics  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_keys(specs: list[dict[str, Any]]) -> list[str]:
    return [key for spec in specs if (key := canonical_key(spec)) is not None]


def weighted_mean(rows: list[dict[str, Any]], metric: str, populations: dict[str, int]) -> float:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row["gold_family"])].append(row)
    total = sum(populations.values())
    return sum(populations[name] * mean(float(row[metric]) for row in strata[name]) for name in populations) / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    design = json.loads(args.design.read_text(encoding="utf-8"))
    populations = {str(key): int(value) for key, value in design["population_strata_after_screen"].items()}
    gold = {row["record_id"]: dedupe_specs(row["gold_answer"]) for row in load_split(args.data_dir, "dev")}
    files = sorted(args.new_run_dir.glob("*.jsonl"))
    files.extend(sorted(args.baseline_dir.glob("*_91_all_taf_eligible_full.jsonl")))
    per_case: list[dict[str, Any]] = []
    run_lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for path in files:
        for run in read_jsonl(path):
            permutation = int(run.get("permutation_seed", run.get("seed", 91)))
            model = str(run["model"])
            record_id = str(run["record_id"])
            pool = dedupe_specs(run.get("pool", []))
            llm = dedupe_specs(run.get("llm_top5", []))[:5]
            rrf = dedupe_specs(run.get("rrf_top5", []))[:5]
            golds = gold[record_id]
            llm_metrics = ranked_exact_metrics(llm, golds, (1, 5))
            rrf_metrics = ranked_exact_metrics(rrf, golds, (1, 5))
            top_key = candidate_keys(llm[:1])
            presented = list(run.get("presentation_order", []))
            selected_id = str(run.get("llm_ranked_ids", [""])[0]) if run.get("llm_ranked_ids") else ""
            per_case.append({
                "record_id": record_id,
                "model": model,
                "permutation_seed": permutation,
                "gold_family": "+".join(sorted({str(item.get("mark", "unknown")) for item in golds})),
                "pool_size": len(pool),
                "pool_signature": "|".join(candidate_keys(pool)),
                "top1_key": top_key[0] if top_key else "",
                "top5_keys": "|".join(candidate_keys(llm)),
                "llm_hit1": llm_metrics["Hit@1"],
                "llm_hit5": llm_metrics["Hit@5"],
                "rrf_hit1": rrf_metrics["Hit@1"],
                "prefix_complete": float(bool(run.get("prefix_complete"))),
                "selected_presentation_position": presented.index(selected_id) + 1 if selected_id in presented else 0,
            })
            run_lookup[(model, permutation, record_id)] = run

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_case:
        groups[(str(row["model"]), int(row["permutation_seed"]))].append(row)
    permutation_summary: list[dict[str, Any]] = []
    for (model, permutation), rows in sorted(groups.items()):
        if len(rows) != 150:
            raise SystemExit(f"{model} permutation {permutation}: expected 150 rows, found {len(rows)}")
        hit1 = weighted_mean(rows, "llm_hit1", populations)
        rrf = weighted_mean(rows, "rrf_hit1", populations)
        permutation_summary.append({
            "model": model,
            "permutation_seed": permutation,
            "cases": len(rows),
            "hit1": hit1,
            "hit5": weighted_mean(rows, "llm_hit5", populations),
            "rrf_hit1": rrf,
            "llm_minus_rrf_hit1": hit1 - rrf,
            "prefix_complete": weighted_mean(rows, "prefix_complete", populations),
            "mean_selected_presentation_position": weighted_mean(rows, "selected_presentation_position", populations),
        })

    pairwise: list[dict[str, Any]] = []
    models = sorted({str(row["model"]) for row in per_case})
    permutations = sorted({int(row["permutation_seed"]) for row in per_case})
    analyzed_record_ids = sorted({str(row["record_id"]) for row in per_case})
    for model in models:
        for index, left_seed in enumerate(permutations):
            for right_seed in permutations[index + 1:]:
                stability_rows: list[dict[str, Any]] = []
                for record_id in analyzed_record_ids:
                    left = run_lookup[(model, left_seed, record_id)]
                    right = run_lookup[(model, right_seed, record_id)]
                    left_pool, right_pool = candidate_keys(left.get("pool", [])), candidate_keys(right.get("pool", []))
                    if left_pool != right_pool:
                        raise SystemExit(f"pool changed for {model} {record_id}")
                    left_keys, right_keys = candidate_keys(left.get("llm_top5", [])), candidate_keys(right.get("llm_top5", []))
                    union = set(left_keys) | set(right_keys)
                    stability_rows.append({
                        "gold_family": "+".join(sorted({str(item.get("mark", "unknown")) for item in gold[record_id]})),
                        "top1_identity": float(bool(left_keys) and bool(right_keys) and left_keys[0] == right_keys[0]),
                        "top5_jaccard": len(set(left_keys) & set(right_keys)) / len(union) if union else 1.0,
                    })
                pairwise.append({
                    "model": model,
                    "left_permutation": left_seed,
                    "right_permutation": right_seed,
                    "top1_identity": weighted_mean(stability_rows, "top1_identity", populations),
                    "top5_jaccard": weighted_mean(stability_rows, "top5_jaccard", populations),
                })

    model_summary: list[dict[str, Any]] = []
    for model in models:
        rows = [row for row in permutation_summary if row["model"] == model]
        unanimous_rows: list[dict[str, Any]] = []
        for record_id in analyzed_record_ids:
            keys = []
            for permutation in permutations:
                run = run_lookup[(model, permutation, record_id)]
                top = candidate_keys(run.get("llm_top5", [])[:1])
                keys.append(top[0] if top else "")
            unanimous_rows.append({
                "gold_family": "+".join(sorted({str(item.get("mark", "unknown")) for item in gold[record_id]})),
                "unanimous": float(len(set(keys)) == 1),
            })
        pair_rows = [row for row in pairwise if row["model"] == model]
        model_summary.append({
            "model": model,
            "permutations": len(permutations),
            "hit1_min": min(float(row["hit1"]) for row in rows),
            "hit1_max": max(float(row["hit1"]) for row in rows),
            "hit5_min": min(float(row["hit5"]) for row in rows),
            "hit5_max": max(float(row["hit5"]) for row in rows),
            "llm_minus_rrf_hit1_min": min(float(row["llm_minus_rrf_hit1"]) for row in rows),
            "llm_minus_rrf_hit1_max": max(float(row["llm_minus_rrf_hit1"]) for row in rows),
            "mean_pairwise_top1_identity": mean(float(row["top1_identity"]) for row in pair_rows),
            "mean_pairwise_top5_jaccard": mean(float(row["top5_jaccard"]) for row in pair_rows),
            "unanimous_top1_identity": weighted_mean(unanimous_rows, "unanimous", populations),
            "prefix_complete_min": min(float(row["prefix_complete"]) for row in rows),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case.csv", per_case)
    write_csv(args.output_dir / "permutation_summary.csv", permutation_summary)
    write_csv(args.output_dir / "pairwise_stability.csv", pairwise)
    write_csv(args.output_dir / "model_summary.csv", model_summary)
    manifest = {
        "status": "complete",
        "models": models,
        "permutations": permutations,
        "case_runs": len(per_case),
        "pool_invariant": "canonical eligible pool and RRF order identical across presentation permutations",
        "decoding": "temperature 0, seed 91; only candidate presentation permutation changes",
        "output_contract": "exactly min(5,pool_size) distinct IDs",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest, "model_summary": model_summary}, indent=2))


if __name__ == "__main__":
    main()
