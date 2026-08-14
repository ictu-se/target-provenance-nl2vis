from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from audit_legacy import DEFAULT_DATA, DEFAULT_LEGACY, load_legacy_module


HERE = Path(__file__).resolve().parent
MODES = ["full"] + [f"leave_step_{index}_out" for index in range(1, 7)]


def install_step4_ablation(legacy: Any) -> None:
    original = legacy.clone_item_with_step_ablation

    def extended(item: dict[str, Any], step_mode: str) -> dict[str, Any]:
        if step_mode != "leave_step_4_out":
            return original(item, step_mode)
        cloned = deepcopy(item)
        cloned["steps"]["step_4"]["answer"] = []
        return cloned

    legacy.clone_item_with_step_ablation = extended


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path, default=HERE / "out" / "a" / "step_ablation")
    args = parser.parse_args()

    legacy = load_legacy_module(args.legacy_dir, args.data_dir)
    install_step4_ablation(legacy)
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        training = legacy.build_training_rows_with_ablation(
            split="train", max_candidates=5, ablation_mode="full", step_mode=mode
        )
        vectorizer = DictVectorizer(sparse=True)
        matrix = vectorizer.fit_transform(row["features"] for row in training)
        labels = [row["label"] for row in training]
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42
        )
        model.fit(matrix, labels)
        heuristic = legacy.run_split("test", max_candidates=5, step_mode=mode)["summary"]
        learned = legacy.run_split_with_model(
            "test", vectorizer, model, max_candidates=5, ablation_mode="full", step_mode=mode
        )["summary"]
        oracle = legacy.run_oracle_split("test", max_candidates=5, step_mode=mode)["summary"]
        for ranker, result in (("heuristic", heuristic), ("learned", learned), ("oracle", oracle)):
            rows.append({
                "step_mode": mode, "ranker": ranker, "test_n": result["count"],
                "Hit@1": result["Hit@1"], "Hit@3": result["Hit@3"], "Hit@5": result["Hit@5"],
                "MRR": result["MRR"], "avg_candidate_count": result["avg_candidate_count"],
                "training_rows": len(training) if ranker == "learned" else "",
                "training_positive_rate": sum(labels) / len(labels) if ranker == "learned" else "",
                "feature_count": len(vectorizer.feature_names_) if ranker == "learned" else "",
            })
        print(f"Completed {mode}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(), "modes": MODES,
            "policy": "Each learned reranker is retrained on the train split under the same answer-field ablation used at test time.",
            "step4_extension": "The untouched legacy module lacked a step-4 switch; the audit wrapper deep-copies each record and empties only step_4.answer before calling the unchanged pipeline.",
            "reasoning_use": "The constructor and rankers consume step answers, not step reasoning strings.",
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
