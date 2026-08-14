from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from audit_legacy import DEFAULT_DATA, DEFAULT_LEGACY, load_legacy_module, train_legacy_reranker
from common import ranked_exact_metrics


HERE = Path(__file__).resolve().parent


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path, default=HERE / "out" / "a" / "legacy_pair")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    legacy = load_legacy_module(args.legacy_dir, args.data_dir)
    vectorizer, model, _ = train_legacy_reranker(legacy)
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(legacy.load_split("test")):
        heuristic = legacy.rank_candidates(item, max_candidates=5, step_mode="full")
        learned = legacy.rank_candidates_with_model(item, vectorizer, model, max_candidates=5, step_mode="full")
        h = ranked_exact_metrics(heuristic, item["gold_answer"], (1, 3, 5))
        l = ranked_exact_metrics(learned, item["gold_answer"], (1, 3, 5))
        steps = item["steps"]
        step6 = steps.get("step_6", {}).get("answer", [])
        gold_keys = {legacy.canonical_key(spec) for spec in item["gold_answer"]}
        step6_keys = {legacy.canonical_key(spec) for spec in step6} if isinstance(step6, list) else set()
        cases.append({
            "record_id": f"nvbench2:test:{index}", "step6_set_equal_gold": float(step6_keys == gold_keys),
            **{f"heuristic_{key.lower()}": value for key, value in h.items()},
            **{f"learned_{key.lower()}": value for key, value in l.items()},
        })

    rng = random.Random(20260807)
    summary: list[dict[str, Any]] = []
    for metric in ("hit@1", "hit@3", "hit@5", "mrr"):
        differences = [case[f"learned_{metric}"] - case[f"heuristic_{metric}"] for case in cases]
        estimates = [mean(rng.choices(differences, k=len(differences))) for _ in range(args.bootstrap)]
        row: dict[str, Any] = {
            "metric": metric, "n": len(cases),
            "heuristic": mean(case[f"heuristic_{metric}"] for case in cases),
            "learned": mean(case[f"learned_{metric}"] for case in cases),
            "paired_difference": mean(differences),
            "paired_ci_low": percentile(estimates, 0.025), "paired_ci_high": percentile(estimates, 0.975),
        }
        if metric.startswith("hit"):
            h_only = sum(case[f"heuristic_{metric}"] > case[f"learned_{metric}"] for case in cases)
            l_only = sum(case[f"learned_{metric}"] > case[f"heuristic_{metric}"] for case in cases)
            row.update({"heuristic_only": h_only, "learned_only": l_only, "mcnemar_exact_p": exact_mcnemar_p(h_only, l_only)})
        summary.append(row)

    leakage_groups: list[dict[str, Any]] = []
    for equal in (0.0, 1.0):
        group = [case for case in cases if case["step6_set_equal_gold"] == equal]
        leakage_groups.append({
            "step6_set_equal_gold": int(equal), "n": len(group),
            "heuristic_hit1": mean(case["heuristic_hit@1"] for case in group),
            "learned_hit1": mean(case["learned_hit@1"] for case in group),
            "learned_hit5": mean(case["learned_hit@5"] for case in group),
            "learned_mrr": mean(case["learned_mrr"] for case in group),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("paired_summary.csv", summary), ("by_step6_equality.csv", leakage_groups), ("per_case.csv", cases)):
        with (args.output_dir / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"bootstrap": args.bootstrap, "bootstrap_seed": 20260807, "test_n": len(cases)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"paired": summary, "by_step6_equality": leakage_groups}, indent=2))


if __name__ == "__main__":
    main()
