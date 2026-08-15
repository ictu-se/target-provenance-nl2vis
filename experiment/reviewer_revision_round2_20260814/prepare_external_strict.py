from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
if not BASE.exists():
    BASE = HERE.parent
sys.path.insert(0, str(BASE))

from common import sha256_file  # noqa: E402
from prepare_nvbench_v1_external import (  # noqa: E402
    GROUPED,
    MARKS,
    canonicalize_gold_fields,
    clean_field,
    gold_fields,
    parse_measure,
    schema_for,
    split_csv,
)
from run_forward_ollama import validation_errors  # noqa: E402


def strict_gold(item: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    sql = str(item["vis_query"]["data_part"]["sql_part"])
    select_match = re.search(r"(?is)\bselect\s+(.+?)\s+from\s+", sql)
    if not select_match:
        return None, "missing_select"
    selected = split_csv(select_match.group(1))
    if len(selected) != 2:
        return None, "nonbinary_select"
    x_field = clean_field(selected[0])
    y = parse_measure(selected[1])
    if not x_field or y is None:
        return None, "measure_parse_failure"
    vis_x = clean_field(str(item.get("vis_obj", {}).get("x_name", "")))
    if vis_x and vis_x.lower() != x_field.lower():
        return None, "sql_visualization_x_mismatch"

    chart = str(item["chart"])
    mark = MARKS[chart]
    if mark == "arc":
        encoding: dict[str, Any] = {"color": {"field": x_field}, "theta": y}
    else:
        encoding = {"x": {"field": x_field}, "y": y}

    if chart in GROUPED:
        group_match = re.search(r"(?is)\bgroup\s+by\s+(.+?)(?:\s+order\s+by\b|$)", sql)
        if not group_match:
            return None, "missing_group_field"
        groups = [clean_field(value) for value in split_csv(group_match.group(1))]
        color = next((value for value in groups if value.lower() != x_field.lower()), "")
        if not color:
            return None, "missing_distinct_group_field"
        encoding["color"] = {"field": color}

    binning = str(item.get("vis_query", {}).get("data_part", {}).get("binning", "") or "").strip()
    if binning:
        bin_match = re.fullmatch(r"(?i)BIN\s+(.+?)\s+BY\s+(YEAR|MONTH|WEEKDAY)", binning)
        if not bin_match:
            return None, "unsupported_binning"
        bin_field, interval = bin_match.groups()
        if clean_field(bin_field).lower() != x_field.lower():
            return None, "binning_field_mismatch"
        channel = "color" if mark == "arc" else "x"
        time_unit = {"YEAR": "year", "MONTH": "month", "WEEKDAY": "day"}[interval.upper()]
        encoding[channel]["timeUnit"] = time_unit

    if re.search(r"(?i)\border\s+by\b", sql):
        order_match = re.search(r"(?is)\border\s+by\s+(.+?)\s+(asc|desc)\s*$", sql)
        if not order_match:
            return None, "unsupported_order_syntax"
        order_expression, direction = order_match.groups()
        compact = lambda value: re.sub(r"[^a-z0-9*]", "", value.lower().replace("t1.", "").replace("t2.", "").replace("t3.", ""))
        order_key = compact(order_expression)
        y_keys = {compact(selected[1]), compact(str(item.get("vis_obj", {}).get("y_name", "")))}
        if order_key not in y_keys:
            return None, "unsupported_order_target"
        descending = direction.lower() == "desc"
        if mark == "arc":
            encoding["color"]["sort"] = "-theta" if descending else "theta"
        else:
            encoding["x"]["sort"] = "-y" if descending else "y"
    return {"mark": mark, "encoding": encoding}, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "external_v3")
    parser.add_argument("--per-family", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    source_json = args.source / "NVBench.json"
    database_dir = args.source / "databases" / "database"
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions = Counter()
    source_hardness = Counter()
    eligible_hardness = Counter()

    for source_id, item in payload.items():
        hardness = str(item.get("hardness", "unknown") or "unknown")
        source_hardness[hardness] += 1
        chart = item.get("chart")
        sql = str(item.get("vis_query", {}).get("data_part", {}).get("sql_part", ""))
        binning = str(item.get("vis_query", {}).get("data_part", {}).get("binning", "") or "").strip()
        if chart not in MARKS:
            exclusions["unsupported_chart"] += 1
            continue
        if re.search(r"(?i)\bdistinct\b", sql):
            exclusions["unsupported_distinct"] += 1
            continue
        if re.search(r"(?i)\b(?:min|max)\s*\(", sql):
            exclusions["unsupported_aggregate"] += 1
            continue
        if re.search(r"(?i)\bwhere\b|\bbin\b|\bhaving\b|\bunion\b|\bintersect\b|\bexcept\b", sql):
            exclusions["unsupported_transform"] += 1
            continue
        if re.search(r"(?i)\blimit\b|\boffset\b", sql):
            exclusions["unsupported_limit_or_offset"] += 1
            continue
        gold, reason = strict_gold(item)
        if gold is None:
            exclusions[reason or "gold_parse_failure"] += 1
            continue
        db_id = str(item.get("db_id", ""))
        nested = database_dir / db_id / f"{db_id}.sqlite"
        flat = database_dir / f"{db_id}.sqlite"
        db_path = nested if nested.exists() and nested.stat().st_size else flat
        schema = schema_for(db_path, sql, gold_fields(gold))
        if schema is None:
            exclusions["schema_failure"] += 1
            continue
        canonicalize_gold_fields(gold, schema)
        errors = validation_errors(gold, {str(column) for column in schema["table_columns"]})
        if errors:
            exclusions["validator_rejection"] += 1
            continue
        queries = [" ".join(str(query).split()) for query in item.get("nl_queries", []) if str(query).strip()]
        if not queries:
            exclusions["missing_query"] += 1
            continue
        eligible_hardness[hardness] += 1
        eligible[str(chart)].append(
            {
                "record_id": f"nvbench-v1-strict:{source_id}",
                "source_benchmark": "nvBench-v1",
                "chart_family": chart,
                "db_id": db_id,
                "csv_file": f"{db_id}.sqlite",
                "nl_query": queries[0],
                "table_schema": schema,
                "steps": {},
                "gold_answer": [gold],
                "source_sql": sql,
                "source_hardness": hardness,
                "adapter_version": "strict-v3",
            }
        )

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for chart in MARKS:
        candidates = sorted(eligible[chart], key=lambda row: row["record_id"])
        if len(candidates) < args.per_family:
            raise RuntimeError(f"Insufficient eligible {chart} records: {len(candidates)}")
        rng.shuffle(candidates)
        selected.extend(candidates[: args.per_family])
    selected.sort(key=lambda row: (list(MARKS).index(str(row["chart_family"])), str(row["record_id"])))

    args.output.mkdir(parents=True, exist_ok=True)
    test_path = args.output / "test.json"
    test_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "adapter_version": "strict-v3",
        "source": str(source_json.resolve()),
        "source_sha256": sha256_file(source_json),
        "source_records": len(payload),
        "source_databases": len({str(item.get("db_id", "")) for item in payload.values()}),
        "eligible_databases": len({str(row["db_id"]) for rows in eligible.values() for row in rows}),
        "seed": args.seed,
        "per_family": args.per_family,
        "selection_count": len(selected),
        "selected_gold_validator_accept": len(selected),
        "selected_gold_within_registered_aggregate_timeunit_vocabulary": len(selected),
        "selection_databases": len({row["db_id"] for row in selected}),
        "selection_by_chart": dict(Counter(str(row["chart_family"]) for row in selected)),
        "eligible_by_chart": {chart: len(rows) for chart, rows in eligible.items()},
        "source_hardness": dict(source_hardness),
        "eligible_hardness": dict(eligible_hardness),
        "selection_hardness": dict(Counter(str(row["source_hardness"]) for row in selected)),
        "exclusions": dict(exclusions),
        "selection_sha256": sha256_file(test_path),
        "scope": "balanced seven-family estimate for the strict canonical-grammar-eligible subset; filters, distinct/min/max aggregation, limits, offsets, unsupported sorting, and unsupported binning are excluded; YEAR, MONTH, and WEEKDAY binning are retained as explicit time units; not nvBench-v1-wide performance",
    }
    (args.output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
