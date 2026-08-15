#!/usr/bin/env python3
"""Audit lineage overlap between nvBench v1 and the committed nvBench 2.0 splits."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROUND2 = HERE.parent / "reviewer_revision_round2_20260814"
BASE = HERE.parent / "reviewer_revision_20260807"
sys.path.insert(0, str(ROUND2))
sys.path.insert(0, str(BASE))

from common import spec_components, stable_json  # noqa: E402
from prepare_external_v2 import MARKS, strict_gold  # noqa: E402
from prepare_nvbench_v1_external import schema_for  # noqa: E402


def normalize_query(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def schema_signature(schema: Any) -> tuple[str, ...]:
    if isinstance(schema, str):
        schema = json.loads(schema)
    return tuple(sorted(str(value).lower() for value in schema.get("table_columns", [])))


def analytic_signature(spec: dict[str, Any]) -> str:
    components = spec_components(spec)
    return stable_json({name: sorted(values) for name, values in components.items()})


def v2_db_id(row: dict[str, Any]) -> str:
    return str(row.get("csv_file", "")).split("@", 1)[0].removesuffix(".csv").removesuffix(".sqlite")


def v1_db_path(database_dir: Path, db_id: str) -> Path:
    nested = database_dir / db_id / f"{db_id}.sqlite"
    flat = database_dir / f"{db_id}.sqlite"
    return nested if nested.exists() and nested.stat().st_size else flat


def strict_signature(item: dict[str, Any]) -> str | None:
    chart = item.get("chart")
    sql = str(item.get("vis_query", {}).get("data_part", {}).get("sql_part", ""))
    if chart not in MARKS:
        return None
    if re.search(r"(?i)\bdistinct\b", sql):
        return None
    if re.search(r"(?i)\b(?:min|max)\s*\(", sql):
        return None
    if re.search(r"(?i)\bwhere\b|\bbin\b|\bhaving\b|\bunion\b|\bintersect\b|\bexcept\b", sql):
        return None
    if re.search(r"(?i)\blimit\b|\boffset\b", sql):
        return None
    gold, _ = strict_gold(item)
    return analytic_signature(gold) if gold is not None else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir", type=Path, required=True)
    parser.add_argument("--v2-data-dir", type=Path, required=True)
    parser.add_argument("--selected-v1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    v1 = json.loads((args.v1_dir / "NVBench.json").read_text(encoding="utf-8"))
    v2_rows: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        rows = json.loads((args.v2_data_dir / f"{split}.json").read_text(encoding="utf-8"))
        for index, row in enumerate(rows):
            row = dict(row)
            row["_split"] = split
            row["_index"] = index
            v2_rows.append(row)
    selected = json.loads(args.selected_v1.read_text(encoding="utf-8"))

    database_dir = args.v1_dir / "databases" / "database"
    v1_db_ids: set[str] = set()
    v1_queries: set[str] = set()
    v1_db_queries: set[tuple[str, str]] = set()
    v1_schemas: set[tuple[str, tuple[str, ...]]] = set()
    v1_analytic: set[tuple[str, str]] = set()
    v1_feature_rows: list[dict[str, Any]] = []
    schema_failures = 0
    for source_id, item in v1.items():
        db_id = str(item.get("db_id", ""))
        v1_db_ids.add(db_id)
        queries = {normalize_query(str(query)) for query in item.get("nl_queries", []) if str(query).strip()}
        v1_queries.update(queries)
        v1_db_queries.update((db_id, query) for query in queries)
        sql = str(item.get("vis_query", {}).get("data_part", {}).get("sql_part", ""))
        schema = schema_for(v1_db_path(database_dir, db_id), sql, set())
        schema_key = schema_signature(schema) if schema else ()
        if schema_key:
            v1_schemas.add((db_id, schema_key))
        else:
            schema_failures += 1
        analytic = strict_signature(item)
        if analytic is not None:
            v1_analytic.add((db_id, analytic))
        v1_feature_rows.append({
            "source_id": source_id,
            "db_id": db_id,
            "has_schema": bool(schema_key),
            "strict_analytic_signature": analytic is not None,
        })

    v2_query_keys = {normalize_query(str(row.get("nl_query", ""))) for row in v2_rows}
    v2_db_ids = {v2_db_id(row) for row in v2_rows}
    v2_db_queries = {(v2_db_id(row), normalize_query(str(row.get("nl_query", "")))) for row in v2_rows}
    v2_schemas = {(v2_db_id(row), schema_signature(row.get("table_schema", {}))) for row in v2_rows}
    v2_analytic = {
        (v2_db_id(row), analytic_signature(gold))
        for row in v2_rows
        for gold in (json.loads(row["gold_answer"]) if isinstance(row.get("gold_answer"), str) else row.get("gold_answer", []))
    }

    def row_overlap(row: dict[str, Any], selected_row: bool = False) -> dict[str, Any]:
        db_id = str(row.get("db_id", "")) if selected_row else v2_db_id(row)
        query = normalize_query(str(row.get("nl_query", "")))
        schema = schema_signature(row.get("table_schema", {}))
        golds = row.get("gold_answer", [])
        if isinstance(golds, str):
            golds = json.loads(golds)
        return {
            "db_id_overlap": db_id in v1_db_ids,
            "query_overlap": query in v1_queries,
            "db_query_overlap": (db_id, query) in v1_db_queries,
            "db_schema_overlap": (db_id, schema) in v1_schemas,
            "db_analytic_overlap": any((db_id, analytic_signature(gold)) in v1_analytic for gold in golds),
        }

    v2_overlap = [row_overlap(row) for row in v2_rows]
    selected_overlap = []
    for row in selected:
        db_id = str(row.get("db_id", ""))
        query = normalize_query(str(row.get("nl_query", "")))
        schema = schema_signature(row.get("table_schema", {}))
        flags = {
            "db_id_overlap": db_id in v2_db_ids,
            "query_overlap": query in v2_query_keys,
            "db_query_overlap": (db_id, query) in v2_db_queries,
            "db_schema_overlap": (db_id, schema) in v2_schemas,
            "db_analytic_overlap": any(
                (db_id, analytic_signature(gold)) in v2_analytic for gold in row.get("gold_answer", [])
            ),
        }
        selected_overlap.append({
            "record_id": row["record_id"],
            "db_id": row["db_id"],
            "chart_family": row["chart_family"],
            **{name: int(value) for name, value in flags.items()},
        })

    def totals(flags: list[dict[str, Any]], denominator: int) -> dict[str, Any]:
        output: dict[str, Any] = {"records": denominator}
        for name in ("db_id_overlap", "query_overlap", "db_query_overlap", "db_schema_overlap", "db_analytic_overlap"):
            count = sum(bool(row[name]) for row in flags)
            output[name] = {"count": count, "proportion": count / denominator}
        return output

    v1_record_db_overlap = sum(str(item.get("db_id", "")) in {v2_db_id(row) for row in v2_rows} for item in v1.values())
    summary = {
        "scope": "descriptive cross-release lineage audit; overlap is not evidence of data independence or external generalization",
        "nvbench_v1": {
            "records": len(v1),
            "databases": len(v1_db_ids),
            "schema_derivation_failures": schema_failures,
            "records_with_database_in_v2": v1_record_db_overlap,
            "records_with_database_in_v2_proportion": v1_record_db_overlap / len(v1),
            "unique_queries": len(v1_queries),
            "strict_vql_analytic_signatures": len(v1_analytic),
        },
        "nvbench_2_0": {
            "records": len(v2_rows),
            "databases": len(v2_db_ids),
            "unique_queries": len(v2_query_keys),
            "overlap_with_v1": totals(v2_overlap, len(v2_rows)),
        },
        "strict_selected_105": {
            "records": len(selected),
            "overlap_with_nvbench_2_0": totals(selected_overlap, len(selected)),
        },
        "unique_key_intersections": {
            "database_ids": len(v1_db_ids & v2_db_ids),
            "normalized_queries": len(v1_queries & v2_query_keys),
            "database_query_pairs": len(v1_db_queries & v2_db_queries),
            "database_schema_pairs": len(v1_schemas & v2_schemas),
            "database_analytic_pairs": len(v1_analytic & v2_analytic),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "selected105_overlap.csv", selected_overlap)
    write_csv(args.output_dir / "v1_feature_availability.csv", v1_feature_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
