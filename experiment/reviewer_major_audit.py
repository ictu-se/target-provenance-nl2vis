from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from common import canonical_key, dedupe_specs, load_split, spec_components
from run_forward_ollama import DEFAULT_DATA, schema_columns, validation_errors
from run_forward_reranker import build_pool


HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_f1_variant(left: set[str], right: set[str], empty_reward: float) -> float:
    if not left and not right:
        return empty_reward
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision, recall = overlap / len(left), overlap / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def similarity(pred: Any, gold: Any, empty_reward: float, weights: dict[str, float]) -> float:
    pc, gc = spec_components(pred), spec_components(gold)
    return sum(weights[name] * set_f1_variant(pc[name], gc[name], empty_reward) for name in weights) / sum(weights.values())


def best_similarity(pred: Any, golds: list[dict[str, Any]], empty_reward: float, weights: dict[str, float]) -> float:
    return max((similarity(pred, gold, empty_reward, weights) for gold in golds), default=0.0)


def gold_recall(preds: Iterable[dict[str, Any]], golds: list[dict[str, Any]], k: int = 5) -> tuple[float, int]:
    gold_keys = {canonical_key(gold) for gold in golds}
    pred_keys = {canonical_key(pred) for pred in list(preds)[:k]}
    recovered = len(gold_keys & pred_keys)
    return (recovered / len(gold_keys) if gold_keys else 0.0, recovered)


def condition_from_name(name: str) -> str:
    if "_r_55_t_rich150" in name:
        return "direct-rich"
    if "_s_55_t_cross150" in name:
        return "staged"
    if "_d_55_t_cross150" in name:
        return "direct-basic"
    return "other"


def model_label(model: str) -> str:
    return model.split(":", 1)[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--forward-dir", type=Path, default=HERE / "out" / "f")
    parser.add_argument("--design", type=Path, default=HERE / "design" / "forward_sample150.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "out" / "major_audit")
    args = parser.parse_args()

    test_rows = load_split(args.data_dir, "test")
    gold_by_id = {row["record_id"]: dedupe_specs(row["gold_answer"]) for row in test_rows}
    row_by_id = {row["record_id"]: row for row in test_rows}
    design = json.loads(args.design.read_text(encoding="utf-8"))
    populations = {name: int(info["population"]) for name, info in design["strata"].items()}
    expected = {name: int(info["sample"]) for name, info in design["strata"].items()}

    gold_validation = Counter()
    total_gold = 0
    invalid_examples = []
    for row in test_rows:
        columns = schema_columns(row)
        for gold in dedupe_specs(row["gold_answer"]):
            total_gold += 1
            errors = validation_errors(gold, columns)
            if errors:
                gold_validation.update(errors)
                if len(invalid_examples) < 10:
                    invalid_examples.append({"record_id": row["record_id"], "errors": errors, "spec": gold})
    validator_summary = {
        "test_cases": len(test_rows), "canonical_gold_specifications": total_gold,
        "accepted_gold_specifications": total_gold - sum(1 for row in test_rows for gold in dedupe_specs(row["gold_answer"]) if validation_errors(gold, schema_columns(row))),
        "error_counts_nonexclusive": dict(gold_validation), "invalid_examples": invalid_examples,
    }
    validator_summary["accepted_percent"] = 100 * validator_summary["accepted_gold_specifications"] / total_gold
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validator_gold_summary.json").write_text(json.dumps(validator_summary, indent=2), encoding="utf-8")

    run_files = sorted(args.forward_dir.glob("*_?_55_t_cross150.jsonl")) + sorted(args.forward_dir.glob("*_r_55_t_rich150.jsonl"))
    case_rows = []
    token_rows = []
    validation_taxonomy = []
    equal = {name: 1.0 for name in ("mark", "channels", "fields", "operations", "filters")}
    emphasized = {"mark": 1.0, "channels": 1.0, "fields": 1.0, "operations": 2.0, "filters": 2.0}
    for path in run_files:
        rows = read_jsonl(path)
        if len(rows) != 150:
            continue
        condition = condition_from_name(path.name)
        for run in rows:
            golds = gold_by_id[run["record_id"]]
            preds = dedupe_specs(run.get("candidates", []))
            recall, recovered = gold_recall(preds, golds)
            first = preds[0] if preds else None
            meta = run.get("ollama_metadata") or {}
            case_rows.append({
                "file": path.name, "model": model_label(run["model"]), "condition": condition,
                "record_id": run["record_id"], "gold_family": "+".join(sorted({str(g.get("mark", "unknown")) for g in golds})),
                "gold_count": len(golds), "gold_recall_5": recall,
                "distinct_gold_recovered_5": recovered,
                "graded_equal": best_similarity(first, golds, 1.0, equal) if first else 0.0,
                "graded_no_empty_reward": best_similarity(first, golds, 0.0, equal) if first else 0.0,
                "graded_operation_filter_weighted": best_similarity(first, golds, 1.0, emphasized) if first else 0.0,
            })
            token_rows.append({
                "model": model_label(run["model"]), "condition": condition, "record_id": run["record_id"],
                "prompt_tokens": meta.get("prompt_eval_count") or 0, "generated_tokens": meta.get("eval_count") or 0,
                "total_tokens": (meta.get("prompt_eval_count") or 0) + (meta.get("eval_count") or 0),
                "elapsed_seconds": run.get("elapsed_seconds") or 0,
            })
            for errors in run.get("candidate_validation_errors", []):
                for error in errors:
                    validation_taxonomy.append({"model": model_label(run["model"]), "condition": condition, "error": error.split(":", 1)[0]})

    write_csv(args.output_dir / "per_case_gold_recall_and_sensitivity.csv", case_rows)
    summaries = []
    for key in sorted({(r["model"], r["condition"]) for r in case_rows}):
        group = [r for r in case_rows if (r["model"], r["condition"]) == key]
        for gold_count in ["all", 1, 2, 3, 4, 5, "multi"]:
            if gold_count == "all": subset = group
            elif gold_count == "multi": subset = [r for r in group if r["gold_count"] >= 2]
            else: subset = [r for r in group if r["gold_count"] == gold_count]
            if not subset: continue
            weighted = gold_count == "all" and set(r["gold_family"] for r in subset) == set(populations)
            def average(metric: str) -> float:
                if not weighted:
                    return mean(r[metric] for r in subset)
                strata = defaultdict(list)
                for row in subset: strata[row["gold_family"]].append(row)
                return sum(populations[name] * mean(r[metric] for r in rows) for name, rows in strata.items()) / sum(populations.values())
            summaries.append({
                "model": key[0], "condition": key[1], "gold_count_group": gold_count, "n": len(subset),
                "estimator": "design-weighted" if weighted else "raw descriptive",
                "mean_gold_recall_5": average("gold_recall_5"),
                "mean_distinct_gold_recovered_5": average("distinct_gold_recovered_5"),
                "graded_equal": average("graded_equal"),
                "graded_no_empty_reward": average("graded_no_empty_reward"),
                "graded_operation_filter_weighted": average("graded_operation_filter_weighted"),
            })
    write_csv(args.output_dir / "gold_recall_and_metric_sensitivity.csv", summaries)

    token_summary=[]
    for key in sorted({(r["model"], r["condition"]) for r in token_rows}):
        group=[r for r in token_rows if (r["model"],r["condition"])==key]
        token_summary.append({"model":key[0],"condition":key[1],"n":len(group),**{field:mean(r[field] for r in group) for field in ("prompt_tokens","generated_tokens","total_tokens","elapsed_seconds")}})
    write_csv(args.output_dir / "token_cost_summary.csv", token_summary)
    counts=Counter((r["model"],r["condition"],r["error"]) for r in validation_taxonomy)
    write_csv(args.output_dir / "candidate_validation_taxonomy.csv", [{"model":k[0],"condition":k[1],"error":k[2],"count":v} for k,v in sorted(counts.items())])

    pool_rows=[]
    pool_sources={
        "direct-only":"*_d_55_t_cross150.jsonl",
        "staged-only":"*_s_55_t_cross150.jsonl",
        "direct+staged":"*_?_55_t_cross150.jsonl",
    }
    for pool_name,pattern in pool_sources.items():
        grouped=defaultdict(list)
        for path in sorted(args.forward_dir.glob(pattern)):
            if path.name.startswith(("deepseek","phi3_5")): continue
            if len(read_jsonl(path)) != 150: continue
            for row in read_jsonl(path): grouped[row["record_id"]].append(row)
        for record_id,rows in grouped.items():
            pool,_=build_pool(rows); golds=gold_by_id[record_id]
            recall,recovered=gold_recall(pool,golds,k=len(pool))
            pool_rows.append({"pool":pool_name,"record_id":record_id,"gold_family":"+".join(sorted({str(g.get("mark","unknown")) for g in golds})),"pool_size":len(pool),"any_gold":float(recovered>0),"gold_recall":recall,"distinct_gold_recovered":recovered,"rrf_hit1":float(gold_recall(pool,golds,1)[1]>0),"rrf_hit5":float(gold_recall(pool,golds,5)[1]>0)})
    write_csv(args.output_dir / "three_pool_rrf_oracles.csv", pool_rows)
    pool_summary=[]
    for pool in pool_sources:
        group=[r for r in pool_rows if r["pool"]==pool]
        strata=defaultdict(list)
        for row in group: strata[row["gold_family"]].append(row)
        def weighted(metric: str) -> float:
            return sum(populations[name]*mean(r[metric] for r in rows) for name,rows in strata.items())/sum(populations.values())
        pool_summary.append({"pool":pool,"n":len(group),"mean_pool_size":weighted("pool_size"),"oracle_any_gold":weighted("any_gold"),"rrf_hit1":weighted("rrf_hit1"),"rrf_hit5":weighted("rrf_hit5"),"mean_gold_recall":weighted("gold_recall"),"mean_distinct_gold_recovered":weighted("distinct_gold_recovered")})
    write_csv(args.output_dir / "three_pool_summary.csv", pool_summary)
    print(json.dumps({"validator":validator_summary,"run_summaries":len(summaries),"pool_summary":pool_summary},indent=2))


if __name__ == "__main__":
    main()
