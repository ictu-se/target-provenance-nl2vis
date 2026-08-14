from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
DATA = Path(os.environ.get(
    "NVBENCH2_TEST_FILE",
    WORKSPACE / "data_benchmarks" / "datasets" / "nvBench-2.0" / "data" / "nvbench2.0" / "test.json",
))
PER_CASE = HERE / "out" / "a" / "full" / "per_case.csv"
OUTPUT = HERE / "out" / "a" / "full" / "by_task_operation.csv"
MANIFEST = HERE / "out" / "a" / "full" / "task_operation_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def operation_groups(golds: list[dict[str, Any]]) -> set[str]:
    groups: set[str] = set()
    for spec in golds:
        encoding = spec.get("encoding", {})
        if isinstance(encoding, dict):
            for channel in encoding.values():
                if not isinstance(channel, dict):
                    continue
                if "aggregate" in channel:
                    groups.add("aggregate")
                if "bin" in channel:
                    groups.add("bin")
                if "timeUnit" in channel:
                    groups.add("time_unit")
                if "sort" in channel:
                    groups.add("sort")
        transforms = spec.get("transform", [])
        if isinstance(transforms, list) and any(
            isinstance(transform, dict) and "filter" in transform for transform in transforms
        ):
            groups.add("filter")
    if not groups:
        groups.add("plain_encoding")
    return groups


def main() -> None:
    records = json.loads(DATA.read_text(encoding="utf-8"))
    groups_by_index: dict[int, set[str]] = {}
    for index, record in enumerate(records):
        golds = json.loads(record["gold_answer"])
        groups_by_index[index] = operation_groups(golds)

    with PER_CASE.open("r", encoding="utf-8", newline="") as handle:
        cases = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for case in cases:
        index = int(case["index"])
        for task_group in sorted(groups_by_index[index]):
            grouped[(case["model"], case["condition"], task_group)].append(case)

    fields = [
        "model",
        "condition",
        "task_group",
        "n",
        "any_valid_candidate",
        "raw_hit@1",
        "raw_hit@5",
        "top1_macro",
        "best5_macro",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            model, condition, task_group = key
            rows = grouped[key]
            writer.writerow(
                {
                    "model": model,
                    "condition": condition,
                    "task_group": task_group,
                    "n": len(rows),
                    "any_valid_candidate": mean(float(row["any_valid_candidate"]) for row in rows),
                    "raw_hit@1": mean(float(row["raw_hit@1"]) for row in rows),
                    "raw_hit@5": mean(float(row["raw_hit@5"]) for row in rows),
                    "top1_macro": mean(float(row["top1_macro"]) for row in rows),
                    "best5_macro": mean(float(row["best5_macro"]) for row in rows),
                }
            )

    MANIFEST.write_text(
        json.dumps(
            {
                "analysis": "multi-label grouping by operations present in any canonical gold specification",
                "groups": ["aggregate", "filter", "bin", "time_unit", "sort", "plain_encoding"],
                "grouping_use": "analysis only; gold information was never provided to forward generation",
                "test_records": len(records),
                "case_rows": len(cases),
                "data_sha256": sha256(DATA),
                "per_case_sha256": sha256(PER_CASE),
                "output_sha256": sha256(OUTPUT),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
