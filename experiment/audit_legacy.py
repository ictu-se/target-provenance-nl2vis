from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from common import canonical_key, load_split, mean_dict, ranked_exact_metrics, sha256_file, stable_json


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
DEFAULT_DATA = Path(os.environ.get(
    "NVBENCH2_DATA_DIR",
    WORKSPACE / "data_benchmarks" / "datasets" / "nvBench-2.0" / "data" / "nvbench2.0",
))
DEFAULT_LEGACY = Path(os.environ.get("INTENTLENS_LEGACY_DIR", HERE / "legacy_source"))


def load_legacy_module(legacy_dir: Path, data_dir: Path) -> Any:
    source = legacy_dir / "paper4_candidates.py"
    spec = importlib.util.spec_from_file_location("legacy_paper4_candidates", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.NVBENCH2_DIR = data_dir
    return module


def key_set(specs: Any) -> set[str]:
    if not isinstance(specs, list):
        return set()
    return {key for spec in specs if (key := canonical_key(spec)) is not None}


def key_list(specs: Any) -> list[str]:
    if not isinstance(specs, list):
        return []
    return [key for spec in specs if (key := canonical_key(spec)) is not None]


def describe(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "mean": 0.0, "median": 0.0, "max": 0}
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {"min": min(ordered), "mean": mean(ordered), "median": median, "max": max(ordered)}


def audit_split(data_dir: Path, split: str, legacy: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_split(data_dir, split)
    legacy_rows = legacy.load_split(split)
    counters: Counter[str] = Counter()
    sizes: dict[str, list[int]] = {
        "gold": [],
        "step_4": [],
        "step_5": [],
        "step_6": [],
        "raw_step_6_5_4": [],
        "expanded": [],
        "deduplicated_complete": [],
        "legacy_pool_cap_15": [],
        "legacy_heuristic_top_5": [],
    }
    detail_rows: list[dict[str, Any]] = []

    for row, legacy_row in zip(rows, legacy_rows, strict=True):
        gold = row["gold_answer"]
        steps = row["steps"]
        step_answers = {
            name: steps.get(name, {}).get("answer", [])
            for name in ("step_1", "step_2", "step_3", "step_4", "step_5", "step_6")
        }
        gold_keys = key_set(gold)
        step4_keys = key_set(step_answers["step_4"])
        step5_keys = key_set(step_answers["step_5"])
        step6_keys = key_set(step_answers["step_6"])

        counters["step6_set_equal_gold"] += step6_keys == gold_keys
        counters["step6_order_equal_gold"] += key_list(step_answers["step_6"]) == key_list(gold)
        counters["step6_contains_all_gold"] += gold_keys.issubset(step6_keys)
        counters["step6_contains_any_gold"] += bool(gold_keys & step6_keys)
        counters["step5_contains_all_gold"] += gold_keys.issubset(step5_keys)
        counters["step5_contains_any_gold"] += bool(gold_keys & step5_keys)
        counters["step4_contains_all_gold"] += gold_keys.issubset(step4_keys)
        counters["step4_contains_any_gold"] += bool(gold_keys & step4_keys)

        raw_specs: list[dict[str, Any]] = []
        for name in ("step_6", "step_5", "step_4"):
            answer = step_answers[name]
            if isinstance(answer, list):
                raw_specs.extend(spec for spec in answer if isinstance(spec, dict))
        expanded: list[dict[str, Any]] = []
        for candidate in raw_specs:
            expanded.extend(legacy.expand_aggregate_alternatives(candidate))
        deduplicated = legacy.dedupe_specs(expanded)
        capped_pool = legacy.build_candidate_pool(legacy_row, max_candidates=15, step_mode="full")
        heuristic_top5 = legacy.rank_candidates(legacy_row, max_candidates=5, step_mode="full")

        complete_keys = key_set(deduplicated)
        capped_keys = key_set(capped_pool)
        heuristic_keys = key_set(heuristic_top5)
        counters["complete_pool_any_gold"] += bool(complete_keys & gold_keys)
        counters["cap15_pool_any_gold"] += bool(capped_keys & gold_keys)
        counters["heuristic_top5_any_gold"] += bool(heuristic_keys & gold_keys)

        sizes["gold"].append(len(gold_keys))
        sizes["step_4"].append(len(step4_keys))
        sizes["step_5"].append(len(step5_keys))
        sizes["step_6"].append(len(step6_keys))
        sizes["raw_step_6_5_4"].append(len(raw_specs))
        sizes["expanded"].append(len(expanded))
        sizes["deduplicated_complete"].append(len(deduplicated))
        sizes["legacy_pool_cap_15"].append(len(capped_pool))
        sizes["legacy_heuristic_top_5"].append(len(heuristic_top5))

        detail_rows.append(
            {
                "record_id": row["record_id"],
                "gold_count": len(gold_keys),
                "step4_count": len(step4_keys),
                "step5_count": len(step5_keys),
                "step6_count": len(step6_keys),
                "raw_count": len(raw_specs),
                "expanded_count": len(expanded),
                "deduplicated_complete_count": len(deduplicated),
                "cap15_count": len(capped_pool),
                "step6_set_equal_gold": int(step6_keys == gold_keys),
                "step6_contains_any_gold": int(bool(step6_keys & gold_keys)),
                "complete_pool_contains_any_gold": int(bool(complete_keys & gold_keys)),
                "heuristic_top5_contains_any_gold": int(bool(heuristic_keys & gold_keys)),
            }
        )

    n = len(rows)
    summary = {
        "split": split,
        "n": n,
        "rates": {name: count / n for name, count in sorted(counters.items())},
        "pool_sizes": {name: describe(values) for name, values in sizes.items()},
    }
    return summary, detail_rows


def train_legacy_reranker(legacy: Any) -> tuple[DictVectorizer, LogisticRegression, dict[str, Any]]:
    training = legacy.build_training_rows_with_ablation(
        split="train", max_candidates=5, ablation_mode="full", step_mode="full"
    )
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(row["features"] for row in training)
    labels = [row["label"] for row in training]
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(matrix, labels)
    return vectorizer, model, {
        "training_rows": len(training),
        "positive_rate": sum(labels) / len(labels),
        "feature_count": len(vectorizer.feature_names_),
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": 42,
    }


def ranked_complete_pool(
    legacy: Any,
    item: dict[str, Any],
    vectorizer: DictVectorizer,
    model: LogisticRegression,
) -> dict[str, list[dict[str, Any]]]:
    candidates = legacy.build_candidate_pool(item, max_candidates=100000, step_mode="full")
    heuristic_rows = [
        (legacy.score_candidate(item, spec, step_mode="full"), stable_json(spec), spec)
        for spec in candidates
    ]
    heuristic_rows.sort(key=lambda row: (-row[0], row[1]))

    features = [legacy.candidate_feature_dict(item, spec, step_mode="full") for spec in candidates]
    probabilities = model.predict_proba(vectorizer.transform(features))[:, 1] if candidates else []
    learned_rows = [
        (
            float(probability),
            legacy.score_candidate(item, spec, step_mode="full"),
            stable_json(spec),
            spec,
        )
        for spec, probability in zip(candidates, probabilities, strict=True)
    ]
    learned_rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
    gold_keys = {legacy.canonical_key(spec) for spec in item["gold_answer"]}
    oracle = sorted(candidates, key=lambda spec: (-(legacy.canonical_key(spec) in gold_keys), stable_json(spec)))
    return {
        "heuristic": [row[2] for row in heuristic_rows],
        "learned": [row[3] for row in learned_rows],
        "oracle": oracle,
    }


def evaluate_rankings(data_dir: Path, legacy: Any, vectorizer: DictVectorizer, model: LogisticRegression) -> dict[str, Any]:
    items = legacy.load_split("test")
    ks = list(range(1, 16))
    accumulated: dict[str, list[dict[str, float]]] = {name: [] for name in ("heuristic", "learned", "oracle")}
    pool_sizes: list[int] = []
    for item in items:
        rankings = ranked_complete_pool(legacy, item, vectorizer, model)
        pool_sizes.append(len(rankings["oracle"]))
        for name, predictions in rankings.items():
            accumulated[name].append(ranked_exact_metrics(predictions, item["gold_answer"], ks))
    return {
        "fixed_complete_pool": True,
        "ks": ks,
        "pool_size": describe(pool_sizes),
        "metrics": {name: mean_dict(rows) for name, rows in accumulated.items()},
    }


def reproduce_legacy_table(legacy: Any, vectorizer: DictVectorizer, model: LogisticRegression) -> dict[str, Any]:
    heuristic = legacy.run_split("test", max_candidates=5, step_mode="full")["summary"]
    learned = legacy.run_split_with_model(
        "test", vectorizer, model, max_candidates=5, ablation_mode="full", step_mode="full"
    )["summary"]
    oracle = legacy.run_oracle_split("test", max_candidates=5, step_mode="full")["summary"]
    return {"heuristic": heuristic, "learned": learned, "oracle": oracle}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "legacy_audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    legacy = load_legacy_module(args.legacy_dir, args.data_dir)
    split_summaries: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        summary, details = audit_split(args.data_dir, split, legacy)
        split_summaries.append(summary)
        detail_rows.extend(details)

    vectorizer, model, model_info = train_legacy_reranker(legacy)
    reproduced = reproduce_legacy_table(legacy, vectorizer, model)
    complete_pool_results = evaluate_rankings(args.data_dir, legacy, vectorizer, model)

    source_text = (args.legacy_dir / "paper4_candidates.py").read_text(encoding="utf-8")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "data_dir": str(args.data_dir.resolve()),
        "legacy_dir": str(args.legacy_dir.resolve()),
        "input_sha256": {
            split: sha256_file(args.data_dir / f"{split}.json") for split in ("train", "dev", "test")
        },
        "legacy_code_sha256": sha256_file(args.legacy_dir / "paper4_candidates.py"),
        "six_fields": {
            "step_1": "column selection and explicit filters",
            "step_2": "explicit aggregation, binning, or sorting",
            "step_3": "chart-mark selection",
            "step_4": "initial channel mapping",
            "step_5": "completion of obligatory/optional channels",
            "step_6": "implicit transformations, filters, and final chart list",
        },
        "provenance": "All six fields are benchmark-provided annotations generated after valid charts in the reverse-generation pipeline; none is a forward prediction by the legacy system.",
        "legacy_validation_stage": {
            "implemented": "def is_valid" in source_text.lower(),
            "finding": "No schema validation or IsValid pruning is called by build_candidate_pool; the manuscript algorithm claimed a validation stage that the code did not implement.",
        },
        "heuristic_specification": {
            "formula": "3 I(mark=step3) + sum_channels[2 I(aggregate=query_pref) + 0.15 I(no aggregate and no query preference and channel in {x,y})] + sort_term + color_term + 0.35|fields intersect certain_step1| + 0.15|fields intersect ambiguous_step1| + sum_step2 aggregate/sort terms - 0.02 UTF8_JSON_character_length(spec)",
            "sort_term": "+1 if requested sort is present, otherwise -0.25; zero when no sort is requested",
            "color_term": "+0.8 if query mentions color and candidate has color, otherwise -0.4; zero when color is not mentioned",
            "step2_aggregate_term": "+1.25 for each step-2 aggregate present in an encoding, otherwise -0.1",
            "step2_sort_term": "+0.4 when a step-2 sort string is literally present in the lower-cased query",
            "normalization": "none",
            "heuristic_tie_break": "ascending canonical JSON string",
            "learned_tie_break": "descending heuristic score, then ascending canonical JSON string",
            "critical_pool_detail": "Both rankers independently rank the first 15 deduplicated candidates (3K for K=5); the heuristic top five is not the learned reranker's input shortlist."
        },
        "split_audits": split_summaries,
        "reranker": model_info,
        "legacy_table_reproduction": reproduced,
        "complete_pool_coverage_curves": complete_pool_results,
    }
    (args.output_dir / "legacy_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "legacy_pool_stages.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
