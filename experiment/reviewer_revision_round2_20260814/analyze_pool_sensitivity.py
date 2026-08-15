from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
if not BASE.exists():
    BASE = HERE.parent
sys.path.insert(0, str(BASE))

from common import canonical_key, dedupe_specs, load_split  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * p
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def rrf_pool(rows: list[dict[str, Any]], constant: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    candidates: dict[str, dict[str, Any]] = {}
    provenance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for rank, candidate in enumerate(row.get("valid_candidates", []), 1):
            key = canonical_key(candidate)
            if key is None:
                continue
            candidates.setdefault(key, candidate)
            provenance[key].append({"model": row["model"], "condition": row["condition"], "rank": rank})
    scored = [
        (sum(1 / (constant + item["rank"]) for item in sources), key, candidates[key])
        for key, sources in provenance.items()
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored], provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "analysis" / "pools")
    args = parser.parse_args()

    design = json.loads(args.design.read_text(encoding="utf-8"))
    population = {name: int(info["population"]) for name, info in design["strata"].items()}
    test = {row["record_id"]: row for row in load_split(args.data_dir, "test")}
    files = {
        "direct-only": sorted(args.forward_dir.glob("*_d_55_t_cross150.jsonl")),
        "staged-only": sorted(args.forward_dir.glob("*_s_55_t_cross150.jsonl")),
    }
    files = {name: [path for path in paths if len(read_jsonl(path)) == 150] for name, paths in files.items()}
    files["direct+staged"] = files["direct-only"] + files["staged-only"]

    per_case: list[dict[str, Any]] = []
    for pool_name, paths in files.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for path in paths:
            for row in read_jsonl(path):
                grouped[str(row["record_id"])].append(row)
        for record_id, rows in sorted(grouped.items()):
            golds = dedupe_specs(test[record_id]["gold_answer"])
            gold_keys = {canonical_key(gold) for gold in golds}
            raw_union = dedupe_specs(candidate for row in rows for candidate in row.get("candidates", []))
            row_base = {
                "pool": pool_name,
                "record_id": record_id,
                "gold_family": "+".join(sorted({str(gold.get("mark", "unknown")) for gold in golds})),
                "raw_unique": len(raw_union),
            }
            for constant in (10, 30, 60, 100):
                ranked, provenance = rrf_pool(rows, constant)
                keys = [canonical_key(candidate) for candidate in ranked]
                per_case.append(
                    {
                        **row_base,
                        "rrf_constant": constant,
                        "valid_unique": len(ranked),
                        "multi_source_candidates": sum(len(sources) > 1 for sources in provenance.values()),
                        "mean_sources_per_candidate": statistics.mean(len(sources) for sources in provenance.values()) if provenance else 0.0,
                        "oracle": float(bool(set(keys) & gold_keys)),
                        "hit1": float(bool(keys) and keys[0] in gold_keys),
                        "hit5": float(bool(set(keys[:5]) & gold_keys)),
                    }
                )

    write_csv(args.output_dir / "per_case_rrf_sensitivity.csv", per_case)
    summaries: list[dict[str, Any]] = []
    for pool_name in files:
        for constant in (10, 30, 60, 100):
            group = [row for row in per_case if row["pool"] == pool_name and row["rrf_constant"] == constant]
            strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in group:
                strata[row["gold_family"]].append(row)

            def weighted(metric: str) -> float:
                return sum(population[name] * statistics.mean(float(row[metric]) for row in rows) for name, rows in strata.items()) / sum(population.values())

            valid_sizes = [float(row["valid_unique"]) for row in group]
            raw_sizes = [float(row["raw_unique"]) for row in group]
            summaries.append(
                {
                    "pool": pool_name,
                    "rrf_constant": constant,
                    "n": len(group),
                    "raw_unique_mean": statistics.mean(raw_sizes),
                    "valid_unique_mean": statistics.mean(valid_sizes),
                    "valid_unique_sd": statistics.pstdev(valid_sizes),
                    "valid_unique_min": min(valid_sizes),
                    "valid_unique_q25": quantile(valid_sizes, 0.25),
                    "valid_unique_median": quantile(valid_sizes, 0.5),
                    "valid_unique_q75": quantile(valid_sizes, 0.75),
                    "valid_unique_max": max(valid_sizes),
                    "multi_source_candidates_mean": statistics.mean(float(row["multi_source_candidates"]) for row in group),
                    "sources_per_candidate_mean": statistics.mean(float(row["mean_sources_per_candidate"]) for row in group),
                    "oracle_design_weighted": weighted("oracle"),
                    "rrf_hit1_design_weighted": weighted("hit1"),
                    "rrf_hit5_design_weighted": weighted("hit5"),
                }
            )
    write_csv(args.output_dir / "summary_rrf_sensitivity.csv", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
