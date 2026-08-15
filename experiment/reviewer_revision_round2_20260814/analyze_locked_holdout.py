#!/usr/bin/env python3
"""Design-weighted analysis of the locked dev holdout.

The analysis separates full canonical equality from a core-specification
equivalence that removes presentation-only metadata. Primary paired contrasts
were locked as direct-rich minus direct and staged minus direct-rich.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
if not BASE.exists():
    BASE = HERE.parent
sys.path.insert(0, str(BASE))

from analyze_forward import evaluate_case  # noqa: E402
from common import canonical_key, dedupe_specs, load_split, spec_components  # noqa: E402
from metric_equivalence import core_key, difference_taxonomy  # noqa: E402


METRICS = (
    "parse_success",
    "any_valid_candidate",
    "all_candidates_valid",
    "candidate_count",
    "valid_candidate_count",
    "raw_hit@1",
    "raw_hit@5",
    "raw_recall@5",
    "top1_macro",
    "best5_macro",
    "elapsed_seconds",
    "core_hit@1",
    "core_hit@5",
    "graded_no_empty_reward",
    "graded_operation_filter_weighted",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * p
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def weighted_mean(
    cases: list[dict[str, Any]], populations: dict[str, int], value: Callable[[dict[str, Any]], float]
) -> float:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        strata[str(case["gold_family"])].append(case)
    denominator = sum(populations.values())
    return sum(populations[name] * mean(value(case) for case in strata[name]) for name in populations) / denominator


def bootstrap_ci(
    cases: list[dict[str, Any]],
    populations: dict[str, int],
    value: Callable[[dict[str, Any]], float],
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        strata[str(case["gold_family"])].append(case)
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sampled = {
            name: rng.choices(strata[name], k=len(strata[name]))
            for name in populations
        }
        estimates.append(
            sum(populations[name] * mean(value(case) for case in sampled[name]) for name in populations)
            / sum(populations.values())
        )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def core_hit(candidates: list[dict[str, Any]], golds: list[dict[str, Any]], k: int) -> float:
    gold_keys = {core_key(gold) for gold in golds}
    return float(any(core_key(candidate) in gold_keys for candidate in candidates[:k]))


def set_f1(left: set[str], right: set[str], empty_reward: float) -> float:
    if not left and not right:
        return empty_reward
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision, recall = overlap / len(left), overlap / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def graded_variant(
    candidate: dict[str, Any] | None,
    golds: list[dict[str, Any]],
    empty_reward: float,
    weights: dict[str, float],
) -> float:
    if candidate is None:
        return 0.0
    pred = spec_components(candidate)
    scores = []
    for gold in golds:
        target = spec_components(gold)
        scores.append(
            sum(weights[name] * set_f1(pred[name], target[name], empty_reward) for name in weights)
            / sum(weights.values())
        )
    return max(scores, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    design = json.loads(args.design.read_text(encoding="utf-8"))
    populations = {str(k): int(v) for k, v in design["population_strata_after_screen"].items()}
    allocation = {str(k): int(v) for k, v in design["allocation"].items()}
    selected_indices = set(int(value) for value in design["indices"])
    gold_by_id = {
        row["record_id"]: row
        for row in load_split(args.data_dir, "dev")
        if int(row["index"]) in selected_indices
    }

    per_case: list[dict[str, Any]] = []
    raw_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    files = []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        rows = read_jsonl(path)
        files.append({"file": path.name, "rows": len(rows)})
        for run in rows:
            gold_row = gold_by_id.get(str(run.get("record_id")))
            if gold_row is None:
                continue
            case = evaluate_case(run, gold_row)
            candidates = dedupe_specs(run.get("candidates", []))[:5]
            golds = dedupe_specs(gold_row["gold_answer"])
            case["core_hit@1"] = core_hit(candidates, golds, 1)
            case["core_hit@5"] = core_hit(candidates, golds, 5)
            equal_weights = {name: 1.0 for name in ("mark", "channels", "fields", "operations", "filters")}
            emphasized_weights = {
                "mark": 1.0,
                "channels": 1.0,
                "fields": 1.0,
                "operations": 2.0,
                "filters": 2.0,
            }
            first = candidates[0] if candidates else None
            case["graded_no_empty_reward"] = graded_variant(first, golds, 0.0, equal_weights)
            case["graded_operation_filter_weighted"] = graded_variant(first, golds, 1.0, emphasized_weights)
            per_case.append(case)
            raw_by_key[(str(run["model"]), str(run["condition"]), str(run["record_id"]))] = run

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in per_case:
        groups[(str(case["model"]), str(case["condition"]))].append(case)

    expected_conditions = {"direct", "direct_rich", "staged"}
    expected_models = sorted({str(case["model"]) for case in per_case})
    errors = []
    for model in expected_models:
        for condition in expected_conditions:
            cases = groups.get((model, condition), [])
            if len(cases) != len(selected_indices):
                errors.append(f"{model}/{condition}: expected {len(selected_indices)}, found {len(cases)}")
            observed = Counter(str(case["gold_family"]) for case in cases)
            if dict(observed) != allocation:
                errors.append(f"{model}/{condition}: stratum allocation mismatch {dict(observed)}")
    if errors:
        raise SystemExit("Incomplete or mismatched locked analysis:\n" + "\n".join(errors))

    summaries = []
    for (model, condition), cases in sorted(groups.items()):
        row: dict[str, Any] = {"model": model, "condition": condition, "n": len(cases)}
        for offset, metric in enumerate(METRICS):
            value = lambda case, name=metric: float(case[name])
            row[metric] = weighted_mean(cases, populations, value)
            low, high = bootstrap_ci(cases, populations, value, args.bootstrap, args.seed + offset)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        summaries.append(row)

    contrasts = []
    taxonomy = []
    for model in expected_models:
        by_condition = {
            condition: {str(case["record_id"]): case for case in groups[(model, condition)]}
            for condition in expected_conditions
        }
        for baseline, comparator, label in (
            ("direct", "direct_rich", "direct-rich minus direct-basic"),
            ("direct_rich", "staged", "staged-rich minus direct-rich"),
            ("direct", "staged", "staged-rich minus direct-basic (secondary)"),
        ):
            paired = []
            for record_id in sorted(by_condition[baseline]):
                left = by_condition[baseline][record_id]
                right = by_condition[comparator][record_id]
                paired.append({**left, **{f"delta_{metric}": float(right[metric]) - float(left[metric]) for metric in METRICS}})

                left_run = raw_by_key[(model, baseline, record_id)]
                right_run = raw_by_key[(model, comparator, record_id)]
                gold = gold_by_id[record_id]
                left_first = (left_run.get("candidates") or [None])[0]
                right_first = (right_run.get("candidates") or [None])[0]
                golds = dedupe_specs(gold["gold_answer"])
                left_exact = any(canonical_key(left_first) == canonical_key(item) for item in golds) if left_first else False
                right_exact = any(canonical_key(right_first) == canonical_key(item) for item in golds) if right_first else False
                if right_exact and not left_exact:
                    matched_gold = next(item for item in golds if canonical_key(right_first) == canonical_key(item))
                    category = difference_taxonomy(left_first, matched_gold)
                    taxonomy.append({
                        "model": model,
                        "contrast": label,
                        "record_id": record_id,
                        "gold_family": left["gold_family"],
                        "baseline_difference": category,
                        "baseline_core_equal_comparator_matched_gold": core_key(left_first) == core_key(matched_gold) if left_first else False,
                        "baseline_top1": json.dumps(left_first, sort_keys=True),
                        "comparator_top1": json.dumps(right_first, sort_keys=True),
                        "golds": json.dumps(golds, sort_keys=True),
                    })

            out: dict[str, Any] = {"model": model, "contrast": label, "n": len(paired)}
            for offset, metric in enumerate(METRICS):
                delta_name = f"delta_{metric}"
                value = lambda case, name=delta_name: float(case[name])
                out[delta_name] = weighted_mean(paired, populations, value)
                low, high = bootstrap_ci(paired, populations, value, args.bootstrap, args.seed + 100 + offset)
                out[f"{delta_name}_ci_low"] = low
                out[f"{delta_name}_ci_high"] = high
            contrasts.append(out)

    taxonomy_summary = [
        {"model": model, "contrast": contrast, "baseline_difference": category, "cases": count}
        for (model, contrast, category), count in sorted(
            Counter((row["model"], row["contrast"], row["baseline_difference"]) for row in taxonomy).items()
        )
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case.csv", per_case)
    write_csv(args.output_dir / "condition_summary.csv", summaries)
    write_csv(args.output_dir / "paired_contrasts.csv", contrasts)
    write_csv(args.output_dir / "exact_gain_taxonomy.csv", taxonomy)
    write_csv(args.output_dir / "exact_gain_taxonomy_summary.csv", taxonomy_summary)
    write_csv(args.output_dir / "file_inventory.csv", files)
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps({
            "input_dir": str(args.input_dir.resolve()),
            "design": str(args.design.resolve()),
            "models": expected_models,
            "conditions": sorted(expected_conditions),
            "n_per_run": len(selected_indices),
            "population_after_screen_exclusion": sum(populations.values()),
            "bootstrap_repetitions": args.bootstrap,
            "bootstrap_seed": args.seed,
            "full_exact_policy": "full normalized specification equality",
            "core_equivalence_policy": "full equality after removing top-level title, description, usermeta, schema, config, dimensions, and encoding axis/legend/title/format; type and scale are retained because they can affect meaning or rendering",
            "primary_contrasts": ["direct-rich minus direct-basic", "staged-rich minus direct-rich"],
            "secondary_contrast": "staged-rich minus direct-basic",
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Analyzed {len(per_case)} case-run rows across {len(groups)} locked runs")


if __name__ == "__main__":
    main()
