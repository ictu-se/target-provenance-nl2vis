#!/usr/bin/env python3
"""Audit locked reranker pool exposure and output handling from retained runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        nonempty = [row for row in rows if int(row["valid_union_count"]) > 0]
        summaries.append(
            {
                "file": path.name,
                "sha256": sha256(path),
                "model": rows[0]["model"],
                "pool": rows[0]["pool_name"],
                "rows": len(rows),
                "mean_pool_size": sum(int(row["valid_union_count"]) for row in rows) / len(rows),
                "maximum_pool_size": max(int(row["valid_union_count"]) for row in rows),
                "pools_above_five": sum(int(row["valid_union_count"]) > 5 for row in rows),
                "empty_pools": len(rows) - len(nonempty),
                "nonempty_parse_success": sum(bool(row.get("parse_success")) for row in nonempty),
                "nonempty_parse_failures": sum(not bool(row.get("parse_success")) for row in nonempty),
                "returned_below_five": sum(len(row.get("llm_ranked_ids", [])) < min(5, int(row["valid_union_count"])) for row in nonempty),
                "returned_unknown_or_duplicate_ids": sum(
                    len(row.get("llm_ranked_ids", [])) != len(set(row.get("llm_ranked_ids", [])))
                    or any(candidate_id not in set(row.get("presentation_order", [])) for candidate_id in row.get("llm_ranked_ids", []))
                    for row in nonempty
                ),
            }
        )
    result = {
        "status": "complete",
        "input_policy": "Each nonempty prompt includes the query, schema, and every pool member as candidate_id plus canonical specification.",
        "presentation_policy": "All pool members are shuffled deterministically by record ID and seed before prompting.",
        "output_policy": "The constrained response contains one to five distinct existing IDs; omitted pool members receive no LLM rank and are not backfilled.",
        "comparison_policy": "RRF, complete-pool oracle, and each LLM use the same full canonical pool; reported Hit@5 uses only the IDs returned by that method.",
        "runs": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
