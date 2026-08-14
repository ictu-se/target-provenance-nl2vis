from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
DEFAULT_SOURCE = WORKSPACE / "data_benchmarks" / "datasets" / "nvBench"
DEFAULT_OUTPUT = HERE / "ext_v1_20260810"

MARKS = {
    "Bar": "bar",
    "Stacked Bar": "bar",
    "Pie": "arc",
    "Line": "line",
    "Grouping Line": "line",
    "Scatter": "point",
    "Grouping Scatter": "point",
}
GROUPED = {"Stacked Bar", "Grouping Line", "Grouping Scatter"}
_SCHEMA_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}


def clean_field(value: str) -> str:
    value = value.strip().strip('`"[]')
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    return value.strip().strip('`"[]')


def split_csv(text: str) -> list[str]:
    output: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        depth += char == "("
        depth -= char == ")"
        if char == "," and depth == 0:
            output.append(text[start:index].strip())
            start = index + 1
    output.append(text[start:].strip())
    return [part for part in output if part]


def parse_measure(expression: str) -> dict[str, Any] | None:
    expression = expression.strip()
    match = re.fullmatch(r"(?i)(count|avg|sum|min|max)\s*\(\s*(\*|[^)]+)\s*\)", expression)
    if not match:
        return {"field": clean_field(expression)}
    operation, field = match.groups()
    operation = {"avg": "mean"}.get(operation.lower(), operation.lower())
    if operation == "count":
        return {"aggregate": "count"}
    return {"field": clean_field(field), "aggregate": operation}


def parse_gold(item: dict[str, Any]) -> dict[str, Any] | None:
    sql = item["vis_query"]["data_part"]["sql_part"]
    match = re.search(r"(?is)\bselect\s+(.+?)\s+from\s+", sql)
    if not match:
        return None
    selected = split_csv(match.group(1))
    if len(selected) != 2:
        return None
    x_field = clean_field(item["vis_obj"].get("x_name", selected[0]))
    y = parse_measure(selected[1])
    if not x_field or not y:
        return None
    mark = MARKS[item["chart"]]
    if mark == "arc":
        encoding: dict[str, Any] = {"color": {"field": x_field}, "theta": y}
    else:
        encoding = {"x": {"field": x_field}, "y": y}
    if item["chart"] in GROUPED:
        group_match = re.search(r"(?is)\bgroup\s+by\s+(.+?)(?:\s+order\s+by\b|$)", sql)
        if not group_match:
            return None
        groups = [clean_field(value) for value in split_csv(group_match.group(1))]
        color = next((value for value in groups if value.lower() != x_field.lower()), "")
        if not color:
            return None
        encoding["color"] = {"field": color}
    order_match = re.search(r"(?is)\border\s+by\s+(.+?)\s+(asc|desc)\s*$", sql)
    if order_match:
        order_expression, direction = order_match.groups()
        compact = lambda value: re.sub(r"[^a-z0-9*]", "", value.lower().replace("t1.", "").replace("t2.", "").replace("t3.", ""))
        order_key = compact(order_expression)
        y_keys = {compact(selected[1]), compact(str(item["vis_obj"].get("y_name", "")))}
        if order_key not in y_keys:
            return None
        descending = direction.lower() == "desc"
        if mark == "arc":
            encoding["color"]["sort"] = "-theta" if descending else "theta"
        else:
            encoding["x"]["sort"] = "-y" if descending else "y"
    return {"mark": mark, "encoding": encoding}


def table_names(sql: str) -> list[str]:
    return [
        match.group(1).strip('`"[]')
        for match in re.finditer(r"(?i)\b(?:from|join)\s+([\w`\"\[\]]+)", sql)
    ]


def schema_for(db_path: Path, sql: str, required_fields: set[str]) -> dict[str, Any] | None:
    if not db_path.exists():
        return None
    tables = tuple(table_names(sql))
    cache_key = (str(db_path), tuple(sorted(table.lower() for table in tables)))
    cached = _SCHEMA_CACHE.get(cache_key)
    if cached is not None:
        if {field.lower() for field in required_fields}.issubset(
            {str(column).lower() for column in cached["table_columns"]}
        ):
            return cached
        return None
    connection = sqlite3.connect(str(db_path))
    try:
        columns: list[str] = []
        examples: dict[str, list[Any]] = {}
        unique_counts: dict[str, int] = {}
        for table in tables:
            safe_table = table.replace('"', '""')
            info = connection.execute(f'PRAGMA table_info("{safe_table}")').fetchall()
            for column_info in info:
                column = str(column_info[1])
                if column in columns:
                    continue
                columns.append(column)
                safe_column = column.replace('"', '""')
                try:
                    values = connection.execute(
                        f'SELECT DISTINCT "{safe_column}" FROM "{safe_table}" WHERE "{safe_column}" IS NOT NULL LIMIT 3'
                    ).fetchall()
                    examples[column] = [row[0] for row in values]
                    unique_counts[column] = int(connection.execute(
                        f'SELECT COUNT(DISTINCT "{safe_column}") FROM "{safe_table}"'
                    ).fetchone()[0])
                except sqlite3.Error:
                    examples[column] = []
        if not {field.lower() for field in required_fields}.issubset({column.lower() for column in columns}):
            return None
        result = {"table_columns": columns, "column_examples": examples, "unique_value_counts": unique_counts}
        _SCHEMA_CACHE[cache_key] = result
        return result
    finally:
        connection.close()


def gold_fields(spec: dict[str, Any]) -> set[str]:
    return {
        str(value["field"])
        for value in spec["encoding"].values()
        if isinstance(value, dict) and isinstance(value.get("field"), str)
    }


def canonicalize_gold_fields(spec: dict[str, Any], schema: dict[str, Any]) -> None:
    canonical = {str(column).lower(): str(column) for column in schema["table_columns"]}
    for value in spec["encoding"].values():
        if isinstance(value, dict) and isinstance(value.get("field"), str):
            value["field"] = canonical[value["field"].lower()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-family", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    source_json = args.source / "NVBench.json"
    database_dir = args.source / "databases" / "database"
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    eligible: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exclusions = Counter()

    for source_id, item in payload.items():
        chart = item.get("chart")
        sql = str(item.get("vis_query", {}).get("data_part", {}).get("sql_part", ""))
        if chart not in MARKS:
            exclusions["unsupported_chart"] += 1
            continue
        if re.search(r"(?i)\bwhere\b|\bbin\b|\bhaving\b|\bunion\b|\bintersect\b|\bexcept\b", sql):
            exclusions["unsupported_transform"] += 1
            continue
        gold = parse_gold(item)
        if gold is None:
            exclusions["gold_parse_failure"] += 1
            continue
        db_id = str(item.get("db_id", ""))
        nested_db = database_dir / db_id / f"{db_id}.sqlite"
        flat_db = database_dir / f"{db_id}.sqlite"
        db_path = nested_db if nested_db.exists() and nested_db.stat().st_size else flat_db
        schema = schema_for(db_path, sql, gold_fields(gold))
        if schema is None:
            exclusions["schema_failure"] += 1
            continue
        canonicalize_gold_fields(gold, schema)
        queries = [" ".join(str(query).split()) for query in item.get("nl_queries", []) if str(query).strip()]
        if not queries:
            exclusions["missing_query"] += 1
            continue
        eligible[chart].append({
            "record_id": f"nvbench-v1:{source_id}",
            "source_benchmark": "nvBench-v1",
            "chart_family": chart,
            "db_id": db_id,
            "csv_file": f"{db_id}.sqlite",
            "nl_query": queries[0],
            "table_schema": schema,
            "steps": {},
            "gold_answer": [gold],
            "source_sql": sql,
            "source_hardness": item.get("hardness", ""),
        })

    randomizer = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for chart in MARKS:
        candidates = sorted(eligible[chart], key=lambda row: row["record_id"])
        randomizer.shuffle(candidates)
        selected.extend(candidates[: args.per_family])
    selected.sort(key=lambda row: (list(MARKS).index(row["chart_family"]), row["record_id"]))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "test.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "source": str(source_json),
        "source_records": len(payload),
        "seed": args.seed,
        "per_family": args.per_family,
        "selection_count": len(selected),
        "selection_by_chart": dict(Counter(row["chart_family"] for row in selected)),
        "eligible_by_chart": {chart: len(rows) for chart, rows in eligible.items()},
        "exclusions": dict(exclusions),
        "scope": "single-gold cross-benchmark transfer; WHERE/BIN/HAVING/set operations and non-canonical ordering excluded",
    }
    (args.output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
