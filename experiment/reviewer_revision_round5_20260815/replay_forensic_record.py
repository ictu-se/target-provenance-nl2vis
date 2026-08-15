#!/usr/bin/env python3
"""Replay headline forensic estimates from the released retained record."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-per-case", type=Path, required=True)
    parser.add_argument("--pool-stages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paired = read_csv(args.paired_per_case)
    pool = [row for row in read_csv(args.pool_stages) if row["record_id"].startswith("nvbench2:test:")]
    if len(paired) != 751 or len(pool) != 751:
        raise SystemExit(f"expected 751 test rows, found paired={len(paired)}, pool={len(pool)}")
    result = {
        "status": "complete",
        "scope": "replay from retained per-case candidate/ranking evidence; not the missing original constructor source",
        "n": len(paired),
        "heuristic": {
            metric: mean(float(row[f"heuristic_{metric}"]) for row in paired)
            for metric in ("hit@1", "hit@3", "hit@5", "mrr")
        },
        "logistic": {
            metric: mean(float(row[f"learned_{metric}"]) for row in paired)
            for metric in ("hit@1", "hit@3", "hit@5", "mrr")
        },
        "step6_set_equal_gold": mean(float(row["step6_set_equal_gold"]) for row in pool),
        "step6_contains_any_gold": mean(float(row["step6_contains_any_gold"]) for row in pool),
        "complete_pool_contains_any_gold": mean(float(row["complete_pool_contains_any_gold"]) for row in pool),
        "heuristic_top5_contains_any_gold": mean(float(row["heuristic_top5_contains_any_gold"]) for row in pool),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
