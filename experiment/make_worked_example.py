from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_legacy import DEFAULT_DATA, DEFAULT_LEGACY, load_legacy_module, train_legacy_reranker
from common import stable_json


HERE = Path(__file__).resolve().parent


def is_gold(legacy: Any, spec: dict[str, Any], gold_keys: set[str]) -> bool:
    return legacy.canonical_key(spec) in gold_keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output", type=Path, default=HERE / "out" / "w" / "e.json")
    args = parser.parse_args()

    legacy = load_legacy_module(args.legacy_dir, args.data_dir)
    vectorizer, model, model_info = train_legacy_reranker(legacy)
    chosen: dict[str, Any] | None = None
    for index, item in enumerate(legacy.load_split("test")):
        pool = legacy.build_candidate_pool(item, max_candidates=100000, step_mode="full")
        gold_keys = {legacy.canonical_key(spec) for spec in item["gold_answer"]}
        heuristic_rows = sorted(
            [(legacy.score_candidate(item, spec, step_mode="full"), stable_json(spec), spec) for spec in pool],
            key=lambda row: (-row[0], row[1]),
        )
        features = [legacy.candidate_feature_dict(item, spec, step_mode="full") for spec in pool]
        probabilities = model.predict_proba(vectorizer.transform(features))[:, 1] if pool else []
        learned_rows = sorted(
            [(float(prob), legacy.score_candidate(item, spec, step_mode="full"), stable_json(spec), spec)
             for spec, prob in zip(pool, probabilities, strict=True)],
            key=lambda row: (-row[0], -row[1], row[2]),
        )
        heuristic_hit5 = any(is_gold(legacy, row[2], gold_keys) for row in heuristic_rows[:5])
        learned_hit1 = bool(learned_rows and is_gold(legacy, learned_rows[0][3], gold_keys))
        if len(pool) > 5 and not heuristic_hit5 and learned_hit1:
            chosen = {
                "index": index,
                "item": item,
                "pool": pool,
                "gold_keys": gold_keys,
                "heuristic_rows": heuristic_rows,
                "learned_rows": learned_rows,
            }
            break
    if chosen is None:
        raise RuntimeError("No worked-example case met the predeclared selection rule")

    item = chosen["item"]
    steps = item["steps"]
    raw_with_sources: list[dict[str, Any]] = []
    expanded_with_sources: list[dict[str, Any]] = []
    for step_name in ("step_6", "step_5", "step_4"):
        answer = steps.get(step_name, {}).get("answer", [])
        if not isinstance(answer, list):
            continue
        for source_rank, spec in enumerate(answer, start=1):
            if not isinstance(spec, dict):
                continue
            raw_with_sources.append({"source_step": step_name, "source_rank": source_rank, "spec": spec})
            for expansion_rank, expanded in enumerate(legacy.expand_aggregate_alternatives(spec), start=1):
                expanded_with_sources.append(
                    {"source_step": step_name, "source_rank": source_rank,
                     "expansion_rank": expansion_rank, "spec": expanded}
                )

    gold_keys = chosen["gold_keys"]
    report = {
        "selection_rule": "first test case with complete pool > 5, heuristic Hit@5=0, and learned Hit@1=1",
        "record_id": f"nvbench2:test:{chosen['index']}",
        "index": chosen["index"],
        "query": item["nl_query"],
        "schema": item["table_schema"],
        "gold_answer": item["gold_answer"],
        "reasoning_fields": {
            name: {"reasoning": steps.get(name, {}).get("reasoning"), "answer": steps.get(name, {}).get("answer")}
            for name in ("step_1", "step_2", "step_3", "step_4", "step_5", "step_6")
        },
        "candidate_stages": {
            "raw_step6_step5_step4": raw_with_sources,
            "raw_count": len(raw_with_sources),
            "expanded": expanded_with_sources,
            "expanded_count": len(expanded_with_sources),
            "validation": {"implemented": False, "removed_count": 0},
            "deduplicated_complete": chosen["pool"],
            "deduplicated_complete_count": len(chosen["pool"]),
        },
        "rankings": {
            "heuristic_formula_location": str((args.legacy_dir / "paper4_candidates.py").resolve()),
            "heuristic": [
                {"rank": rank, "score": score, "is_exact_gold": is_gold(legacy, spec, gold_keys), "spec": spec}
                for rank, (score, _, spec) in enumerate(chosen["heuristic_rows"], start=1)
            ],
            "learned": [
                {"rank": rank, "probability": probability, "heuristic_tiebreak": heuristic,
                 "is_exact_gold": is_gold(legacy, spec, gold_keys), "spec": spec}
                for rank, (probability, heuristic, _, spec) in enumerate(chosen["learned_rows"], start=1)
            ],
        },
        "reranker_training": model_info,
        "interpretation": "This diagnostic succeeds because benchmark-provided step answers, especially step_6.answer, already contain privileged valid-chart information; it is not a deployable query+schema-only result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"record_id": report["record_id"], "raw": len(raw_with_sources),
                      "expanded": len(expanded_with_sources), "deduplicated": len(chosen["pool"])}, indent=2))


if __name__ == "__main__":
    main()
