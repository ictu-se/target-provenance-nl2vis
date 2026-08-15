#!/usr/bin/env python3
"""Run Qwen direct-rich sensitivity to wording and schema serialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
sys.path.insert(0, str(BASE))

from common import dedupe_specs, extract_first_json, load_split, sha256_file, stable_json  # noqa: E402
from run_forward_ollama import (  # noqa: E402
    CHART_RULES,
    DIRECT_RICH_TEMPLATE,
    call_ollama,
    extract_candidates,
    get_model_manifest,
    schema_columns,
    validation_errors,
)


PARAPHRASE_TEMPLATE = """
Act as a ranked NL2Vis specification generator. Infer the analytical request from only the query and supplied schema. Consider grounded columns and filters, operations such as aggregation or temporal grouping, suitable marks, channel assignments, required-channel completion, and necessary transformations. Keep genuinely plausible alternatives, but return no analysis text.

{rules}

USER QUERY
{query}

AVAILABLE SCHEMA
{schema}

Respond exactly as:
{{"candidates":[{{"mark":"...","encoding":{{...}}}}]}}
""".strip()


def schema_bullets(row: dict[str, Any], reverse: bool = False) -> str:
    table_schema = row["table_schema"]
    columns = list(table_schema.get("table_columns", []))
    if reverse:
        columns.reverse()
    examples = table_schema.get("column_examples", {})
    counts = table_schema.get("unique_value_counts", {})
    return "\n".join(
        f"- {column} | examples={stable_json(examples.get(column, []))} | unique_count={counts.get(column, 'unknown')}"
        for column in columns
    )


def prompt_for(row: dict[str, Any], variant: str) -> str:
    schema = schema_bullets(row, reverse=variant == "schema_reverse")
    template = PARAPHRASE_TEMPLATE if variant == "prompt_paraphrase" else DIRECT_RICH_TEMPLATE
    return template.format(rules=CHART_RULES, query=row["nl_query"], schema=schema)


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {str(json.loads(line)["record_id"]) for line in handle if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--variants", nargs="+", default=["prompt_paraphrase", "schema_reverse"])
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    design = json.loads(args.design.read_text(encoding="utf-8"))
    selected = {int(index) for index in design["indices"]}
    rows = [row for row in load_split(args.data_dir, "dev") if int(row["index"]) in selected]
    if len(rows) != len(selected):
        raise SystemExit("locked design indices do not map one-to-one to development rows")
    manifest = get_model_manifest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post-review robustness sensitivity on the existing locked cases",
        "model": {args.model: manifest.get(args.model, {})},
        "variants": args.variants,
        "seed": args.seed,
        "temperature": 0.0,
        "condition": "direct_rich",
        "design_sha256": sha256_file(args.design),
        "data_sha256": sha256_file(args.data_dir / "dev.json"),
        "paraphrase_template_sha256": hashlib.sha256(PARAPHRASE_TEMPLATE.encode()).hexdigest(),
        "baseline_template_sha256": hashlib.sha256(DIRECT_RICH_TEMPLATE.encode()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    for variant in args.variants:
        output = args.output_dir / f"qwen3_14b_direct_rich_{variant}.jsonl"
        done = completed_ids(output)
        with output.open("a", encoding="utf-8") as handle:
            for position, row in enumerate(rows, start=1):
                if row["record_id"] in done:
                    continue
                prompt = prompt_for(row, variant)
                started = time.perf_counter()
                error = None
                response_text = ""
                metadata: dict[str, Any] = {}
                try:
                    response_text, metadata = call_ollama(
                        args.model, prompt, "direct_rich", args.seed, 0.0, args.timeout
                    )
                    parsed = extract_first_json(response_text)
                except Exception as exc:
                    parsed = None
                    error = f"{type(exc).__name__}: {exc}"
                candidates = dedupe_specs(extract_candidates(parsed, "direct_rich"))[:5]
                columns = schema_columns(row)
                errors = [validation_errors(spec, columns) for spec in candidates]
                valid = [spec for spec, item_errors in zip(candidates, errors) if not item_errors]
                record = {
                    "record_id": row["record_id"],
                    "split": "dev",
                    "index": row["index"],
                    "csv_file": row["csv_file"],
                    "model": args.model,
                    "condition": "direct_rich",
                    "robustness_variant": variant,
                    "seed": args.seed,
                    "temperature": 0.0,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "input_policy": "query_and_schema_only",
                    "output_constraint": "shared_json_schema",
                    "parse_success": isinstance(parsed, (dict, list)),
                    "raw_candidate_count": len(extract_candidates(parsed, "direct_rich")),
                    "candidates": candidates,
                    "candidate_validation_errors": errors,
                    "valid_candidates": valid,
                    "parsed_output": parsed if isinstance(parsed, (dict, list)) else None,
                    "trace": None,
                    "response_text": response_text if not isinstance(parsed, (dict, list)) else None,
                    "error": error,
                    "elapsed_seconds": time.perf_counter() - started,
                    "ollama_metadata": metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if position % 10 == 0 or position == len(rows):
                    print(f"{variant}: {position}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
