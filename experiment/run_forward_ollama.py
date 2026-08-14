from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from common import ALLOWED_MARKS, CHANNELS, dedupe_specs, extract_first_json, load_split, normalize_spec, sha256_file, stable_json


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
DEFAULT_DATA = Path(os.environ.get(
    "NVBENCH2_DATA_DIR",
    WORKSPACE / "data_benchmarks" / "datasets" / "nvBench-2.0" / "data" / "nvbench2.0",
))
OLLAMA_URL = "http://127.0.0.1:11434"

CHART_RULES = """
Use only Vega-Lite-like chart specifications with keys mark, encoding, and optional transform.
Allowed marks: bar, line, arc, point, rect, boxplot.
Required channels: bar/line x+y; arc color+theta; point x+y; rect x+y+color; boxplot x+y.
An encoding channel may contain field and optional aggregate (sum, mean, count), bin, timeUnit, or sort.
For row count use {"aggregate":"count"} without a field.
Use only fields in the supplied schema. Add transform filters only when the query requests them.
Return between 1 and 5 distinct candidates ordered from most to least plausible.
Do not use prose outside the requested JSON object.
""".strip()

DIRECT_TEMPLATE = """
You are an NL2Vis candidate generator. The user request may be underspecified, so preserve distinct plausible analytical interpretations.

{rules}

INPUT
Query: {query}
Schema: {schema}

Return exactly:
{{"candidates":[{{"mark":"...","encoding":{{...}}}}]}}
""".strip()

DIRECT_RICH_TEMPLATE = """
You are an NL2Vis candidate generator. Use only the user query and schema below. The user request may be underspecified, so preserve distinct plausible analytical interpretations.

{rules}

Before selecting the final candidates, internally identify the query-grounded columns and explicit filters; explicit aggregation, binning, time-unit, and sorting operations; plausible marks; field-to-channel mappings; missing required channels; and necessary implicit transformations. Return only the final candidates and do not output intermediate stages or reasoning.

INPUT
Query: {query}
Schema: {schema}

Return exactly:
{{"candidates":[{{"mark":"...","encoding":{{...}}}}]}}
""".strip()

STAGED_TEMPLATE = """
You are an NL2Vis candidate generator. Predict a six-stage trace using only the user query and schema below. You do not have benchmark reasoning or gold visualizations.

{rules}

Stage definitions:
1. Extract query-grounded columns and explicit filters.
2. Extract explicit aggregation, binning, time unit, and sorting operations.
3. Select plausible chart marks.
4. Map explicit fields and operations to chart channels.
5. Complete missing required channels with plausible schema fields and preserve alternatives.
6. Add necessary implicit transformations and explicit filters; output the final ranked candidates.

INPUT
Query: {query}
Schema: {schema}

Return exactly one JSON object with keys step_1 through step_6. Each step must contain a concise reasoning string and an answer. The answer for step_6 must be a list of 1 to 5 chart specifications.
""".strip()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def compact_model_slug(value: str) -> str:
    readable = slug(value)[:12]
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:6]
    return f"{readable}_{suffix}"


def prompt_for(row: dict[str, Any], condition: str) -> str:
    table_schema = row["table_schema"]
    examples = table_schema.get("column_examples", {})
    unique_counts = table_schema.get("unique_value_counts", {})
    schema_lines = []
    for column in table_schema.get("table_columns", []):
        schema_lines.append(
            f"- {column} | examples={stable_json(examples.get(column, []))} | unique_count={unique_counts.get(column, 'unknown')}"
        )
    schema = "\n".join(schema_lines)
    template = {
        "direct": DIRECT_TEMPLATE,
        "direct_rich": DIRECT_RICH_TEMPLATE,
        "staged": STAGED_TEMPLATE,
    }[condition]
    return template.format(rules=CHART_RULES, query=row["nl_query"], schema=schema)


def get_model_manifest() -> dict[str, Any]:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=20)
    response.raise_for_status()
    return {item["name"]: item for item in response.json().get("models", [])}


def output_schema(condition: str) -> dict[str, Any]:
    chart = {
        "type": "object",
        "required": ["mark", "encoding"],
        "properties": {
            "mark": {"type": "string", "enum": sorted(ALLOWED_MARKS)},
            "encoding": {"type": "object"},
            "transform": {"type": "array"},
        },
        "additionalProperties": False,
    }
    if condition in {"direct", "direct_rich"}:
        return {
            "type": "object",
            "required": ["candidates"],
            "properties": {
                "candidates": {"type": "array", "minItems": 1, "maxItems": 5, "items": chart}
            },
            "additionalProperties": False,
        }
    step = {
        "type": "object",
        "required": ["reasoning", "answer"],
        "properties": {"reasoning": {"type": "string"}, "answer": {}},
        "additionalProperties": False,
    }
    step6 = {
        "type": "object",
        "required": ["reasoning", "answer"],
        "properties": {
            "reasoning": {"type": "string"},
            "answer": {"type": "array", "minItems": 1, "maxItems": 5, "items": chart},
        },
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "required": [f"step_{index}" for index in range(1, 7)],
        "properties": {**{f"step_{index}": step for index in range(1, 6)}, "step_6": step6},
        "additionalProperties": False,
    }


def call_ollama(model: str, prompt: str, condition: str, seed: int, temperature: float, timeout: int) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": output_schema(condition),
        "think": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": 3072,
            "num_ctx": 8192,
        },
        "keep_alive": "15m",
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            return str(body.get("response", "")), {
                key: body.get(key)
                for key in ("done_reason", "total_duration", "load_duration", "prompt_eval_count", "eval_count", "eval_duration")
            }
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Ollama failed after 3 attempts: {last_error}")


def extract_candidates(parsed: Any, condition: str) -> list[dict[str, Any]]:
    if condition in {"direct", "direct_rich"}:
        if isinstance(parsed, list):
            candidates = parsed
        elif isinstance(parsed, dict):
            candidates = parsed.get("candidates", parsed.get("final_output", []))
        else:
            candidates = []
    else:
        if not isinstance(parsed, dict):
            candidates = []
        else:
            step6 = parsed.get("step_6", {})
            if isinstance(step6, dict):
                candidates = step6.get("answer", step6.get("candidates", []))
            else:
                candidates = step6
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    return [candidate for candidate in candidates if isinstance(candidate, dict)] if isinstance(candidates, list) else []


def schema_columns(row: dict[str, Any]) -> set[str]:
    columns = row["table_schema"].get("table_columns", [])
    return {str(column) for column in columns}


def validation_errors(spec: dict[str, Any], columns: set[str]) -> list[str]:
    errors: list[str] = []
    normalized = normalize_spec(spec) or {}
    mark = normalized.get("mark")
    if mark not in ALLOWED_MARKS:
        errors.append("invalid_mark")
    encoding = normalized.get("encoding", {})
    if not isinstance(encoding, dict):
        return errors + ["missing_encoding"]
    required = {
        "bar": {"x", "y"},
        "line": {"x", "y"},
        "arc": {"color", "theta"},
        "point": {"x", "y"},
        "rect": {"x", "y", "color"},
        "boxplot": {"x", "y"},
    }.get(str(mark), set())
    if not required.issubset(set(encoding)):
        errors.append("missing_required_channel")
    for channel, value in encoding.items():
        if channel not in CHANNELS:
            errors.append(f"unsupported_channel:{channel}")
        if not isinstance(value, dict):
            errors.append(f"non_object_channel:{channel}")
            continue
        field = value.get("field")
        if isinstance(field, str) and field not in columns:
            errors.append(f"unknown_field:{field}")
        if value.get("aggregate") == "count" and field is not None:
            errors.append(f"count_with_field:{channel}")
    for transform in normalized.get("transform", []):
        if not isinstance(transform, dict) or "filter" not in transform:
            errors.append("unsupported_transform")
            continue
        filt = transform.get("filter")
        if not isinstance(filt, dict):
            errors.append("non_object_filter")
        elif isinstance(filt.get("field"), str) and filt["field"] not in columns:
            errors.append(f"unknown_filter_field:{filt['field']}")
    return sorted(set(errors))


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                completed.add(str(json.loads(line)["record_id"]))
            except Exception:
                continue
    return completed


def run_one(
    rows: list[dict[str, Any]],
    model: str,
    condition: str,
    seed: int,
    temperature: float,
    output_path: Path,
    timeout: int,
) -> None:
    done = completed_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for position, row in enumerate(rows, start=1):
            if row["record_id"] in done:
                continue
            prompt = prompt_for(row, condition)
            started = time.perf_counter()
            error = None
            response_text = ""
            metadata: dict[str, Any] = {}
            try:
                response_text, metadata = call_ollama(model, prompt, condition, seed, temperature, timeout)
                parsed = extract_first_json(response_text)
            except Exception as exc:
                parsed = None
                error = f"{type(exc).__name__}: {exc}"
            raw_candidates = extract_candidates(parsed, condition)
            candidates = dedupe_specs(raw_candidates)[:5]
            columns = schema_columns(row)
            candidate_errors = [validation_errors(spec, columns) for spec in candidates]
            valid_candidates = [spec for spec, errors in zip(candidates, candidate_errors) if not errors]
            record = {
                "record_id": row["record_id"],
                "split": row["split"],
                "index": row["index"],
                "csv_file": row["csv_file"],
                "model": model,
                "condition": condition,
                "seed": seed,
                "temperature": temperature,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_policy": "query_and_schema_only",
                "output_constraint": "shared_json_schema",
                "parse_success": isinstance(parsed, (dict, list)),
                "raw_candidate_count": len(raw_candidates),
                "candidates": candidates,
                "candidate_validation_errors": candidate_errors,
                "valid_candidates": valid_candidates,
                "parsed_output": parsed if isinstance(parsed, (dict, list)) else None,
                "trace": parsed if condition == "staged" and isinstance(parsed, dict) else None,
                "response_text": response_text if not isinstance(parsed, (dict, list)) else None,
                "error": error,
                "elapsed_seconds": time.perf_counter() - started,
                "ollama_metadata": metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 10 == 0 or position == len(rows):
                print(f"{model} {condition} seed={seed}: {position}/{len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=["direct", "staged"], choices=("direct", "direct_rich", "staged"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--indices-file", type=Path)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=HERE / "out" / "f")
    args = parser.parse_args()

    rows = load_split(args.data_dir, args.split)
    indices_sha256 = None
    if args.indices_file:
        selected_indices = json.loads(args.indices_file.read_text(encoding="utf-8"))
        if isinstance(selected_indices, dict):
            selected_indices = selected_indices.get("indices", [])
        selected = {int(index) for index in selected_indices}
        rows = [row for row in rows if row["index"] in selected]
        indices_sha256 = sha256_file(args.indices_file)
    else:
        rows = rows[args.offset :]
        if args.limit > 0:
            rows = rows[: args.limit]
    manifest = get_model_manifest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "data_sha256": sha256_file(args.data_dir / f"{args.split}.json"),
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "indices_file": str(args.indices_file.resolve()) if args.indices_file else None,
        "indices_sha256": indices_sha256,
        "run_tag": args.run_tag,
        "models": {model: manifest.get(model, {}) for model in args.models},
        "conditions": args.conditions,
        "seeds": args.seeds,
        "temperature": args.temperature,
        "direct_template_sha256": hashlib.sha256(DIRECT_TEMPLATE.encode("utf-8")).hexdigest(),
        "direct_rich_template_sha256": hashlib.sha256(DIRECT_RICH_TEMPLATE.encode("utf-8")).hexdigest(),
        "staged_template_sha256": hashlib.sha256(STAGED_TEMPLATE.encode("utf-8")).hexdigest(),
        "chart_rules_sha256": hashlib.sha256(CHART_RULES.encode("utf-8")).hexdigest(),
    }
    manifest_text = json.dumps(run_manifest, ensure_ascii=False, indent=2)
    (args.output_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    condition_codes = {"direct": "d", "direct_rich": "r", "staged": "s"}
    condition_code = "".join(condition_codes[condition] for condition in args.conditions)
    seed_code = "-".join(str(seed) for seed in args.seeds)
    manifest_tag = slug(args.run_tag)[:12] if args.run_tag else "untagged"
    immutable_manifest = args.output_dir / f"m_{manifest_tag}_{condition_code}_{seed_code}_{args.split[0]}.json"
    immutable_manifest.write_text(manifest_text, encoding="utf-8")

    for model in args.models:
        for condition in args.conditions:
            for seed in args.seeds:
                tag = f"_{slug(args.run_tag)[:12]}" if args.run_tag else ""
                filename = f"{compact_model_slug(model)}_{condition_codes[condition]}_{seed}_{args.split[0]}{tag}.jsonl"
                run_one(rows, model, condition, seed, args.temperature, args.output_dir / filename, args.timeout)


if __name__ == "__main__":
    main()
