from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from common import canonical_key, dedupe_specs, extract_first_json, load_split, sha256_file, stable_json
from run_forward_ollama import DEFAULT_DATA, OLLAMA_URL, compact_model_slug, get_model_manifest


HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_generator_rows(input_dir: Path, patterns: list[str]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files: list[str] = []
    paths = sorted({path for pattern in patterns for path in input_dir.glob(pattern)})
    for path in paths:
        files.append(path.name)
        for row in read_jsonl(path):
            grouped[str(row["record_id"])].append(row)
    return grouped, files


def build_pool(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[str, dict[str, Any]] = {}
    provenance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for rank, candidate in enumerate(row.get("valid_candidates", []), start=1):
            key = canonical_key(candidate)
            if key is None:
                continue
            by_key.setdefault(key, candidate)
            provenance[key].append(
                {
                    "model": row["model"],
                    "condition": row.get("condition", "unknown"),
                    "source_run": f"{row['model']}::{row.get('condition', 'unknown')}",
                    "rank": rank,
                }
            )

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for key, candidate in by_key.items():
        score = sum(1.0 / (60.0 + item["rank"]) for item in provenance[key])
        scored.append((score, key, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    pool = [candidate for _, _, candidate in scored]
    details = [
        {"canonical_key": key, "rrf_score": score, "sources": provenance[key]}
        for score, key, _ in scored
    ]
    return pool, details


def schema_text(row: dict[str, Any]) -> str:
    schema = row["table_schema"]
    examples = schema.get("column_examples", {})
    counts = schema.get("unique_value_counts", {})
    return "\n".join(
        f"- {column} | examples={stable_json(examples.get(column, []))} | unique_count={counts.get(column, 'unknown')}"
        for column in schema.get("table_columns", [])
    )


def make_prompt(row: dict[str, Any], presented: list[tuple[str, dict[str, Any]]]) -> str:
    candidates = "\n".join(f"{candidate_id}: {stable_json(spec)}" for candidate_id, spec in presented)
    return f"""You are ranking visualization candidates for an underspecified natural-language request.
Use only the query, schema, and supplied candidates. Do not invent or modify candidates.
Rank up to five distinct candidates from most to least defensible. Prefer query fidelity, schema grounding,
valid chart-channel structure, and preservation of genuinely different plausible interpretations.
Do not assume access to benchmark annotations or a gold chart.

Query: {row['nl_query']}
Schema:
{schema_text(row)}

Candidates:
{candidates}

Return only a JSON object with key ranked_ids."""


def call_reranker(model: str, prompt: str, ids: list[str], seed: int, temperature: float, timeout: int) -> tuple[Any, dict[str, Any]]:
    schema = {
        "type": "object",
        "required": ["ranked_ids"],
        "properties": {
            "ranked_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": min(5, len(ids)),
                "uniqueItems": True,
                "items": {"type": "string", "enum": ids},
            }
        },
        "additionalProperties": False,
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "think": False,
        "options": {"temperature": temperature, "seed": seed, "num_predict": 256, "num_ctx": 8192},
        "keep_alive": "15m",
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            parsed = extract_first_json(str(body.get("response", "")))
            metadata = {
                key: body.get(key)
                for key in ("done_reason", "total_duration", "load_duration", "prompt_eval_count", "eval_count", "eval_duration")
            }
            return parsed, metadata
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Reranker failed after 3 attempts: {last_error}")


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row["record_id"]) for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=HERE / "out" / "f")
    parser.add_argument("--input-patterns", nargs="+", default=["*_d_55_t_cross150.jsonl"])
    parser.add_argument("--pool-name", default="direct")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, default=HERE / "out" / "r")
    args = parser.parse_args()

    generator_rows, source_files = load_generator_rows(args.input_dir, args.input_patterns)
    gold_rows = {row["record_id"]: row for row in load_split(args.data_dir, args.split)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_slug = "".join(character if character.isalnum() else "_" for character in args.pool_name.lower()).strip("_")
    output_path = args.output_dir / f"{compact_model_slug(args.model)}_{args.seed}_{pool_slug}.jsonl"
    done = completed_ids(output_path)

    manifest = {
        "model": args.model,
        "model_manifest": get_model_manifest().get(args.model, {}),
        "seed": args.seed,
        "temperature": args.temperature,
        "input_policy": "query_schema_and_valid_forward_candidate_pool_only",
        "source_files": source_files,
        "source_sha256": {name: sha256_file(args.input_dir / name) for name in source_files},
        "input_patterns": args.input_patterns,
        "pool_name": args.pool_name,
        "pool_policy": "canonical union of post-validation candidates",
        "split": args.split,
        "baseline": "RRF score=sum_s 1/(60+source_rank), where each model-condition run is a distinct source; canonical duplicates retain every source contribution; descending score; canonical-key tie break",
        "presentation_policy": "deterministic record-and-seed-specific shuffle to avoid RRF-order leakage",
    }
    (args.output_dir / f"manifest_{compact_model_slug(args.model)}_{args.seed}_{pool_slug}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with output_path.open("a", encoding="utf-8") as handle:
        record_ids = sorted(generator_rows, key=lambda value: int(value.rsplit(":", 1)[-1]))
        for position, record_id in enumerate(record_ids, start=1):
            if record_id in done:
                continue
            row = gold_rows[record_id]
            raw_union = dedupe_specs(
                candidate
                for generator_row in generator_rows[record_id]
                for candidate in generator_row.get("candidates", [])
            )
            pool, pool_details = build_pool(generator_rows[record_id])
            if not pool:
                result = {
                    "record_id": record_id, "index": row["index"], "model": args.model,
                    "pool_name": args.pool_name,
                    "seed": args.seed, "temperature": args.temperature,
                    "raw_union_count": len(raw_union), "valid_union_count": 0,
                    "invalid_unique_removed": len(raw_union), "pool": [], "pool_details": [],
                    "rrf_top5": [], "llm_top5": [], "parse_success": False,
                    "error": "empty_valid_candidate_pool", "elapsed_seconds": 0.0,
                }
            else:
                pairs = [(f"c{index}", spec) for index, spec in enumerate(pool)]
                rng = random.Random(f"{record_id}:{args.seed}")
                presented = list(pairs)
                rng.shuffle(presented)
                prompt = make_prompt(row, presented)
                started = time.perf_counter()
                error = None
                try:
                    parsed, metadata = call_reranker(
                        args.model, prompt, [candidate_id for candidate_id, _ in presented],
                        args.seed, args.temperature, args.timeout,
                    )
                except Exception as exc:
                    parsed, metadata = None, {}
                    error = f"{type(exc).__name__}: {exc}"
                ranked_ids = parsed.get("ranked_ids", []) if isinstance(parsed, dict) else []
                id_to_spec = dict(pairs)
                valid_ranked_ids = []
                for candidate_id in ranked_ids:
                    if candidate_id in id_to_spec and candidate_id not in valid_ranked_ids:
                        valid_ranked_ids.append(candidate_id)
                result = {
                    "record_id": record_id, "index": row["index"], "model": args.model,
                    "pool_name": args.pool_name,
                    "seed": args.seed, "temperature": args.temperature,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "raw_union_count": len(raw_union), "valid_union_count": len(pool),
                    "invalid_unique_removed": len(raw_union) - len(pool),
                    "pool": dedupe_specs(pool), "pool_details": pool_details,
                    "presentation_order": [candidate_id for candidate_id, _ in presented],
                    "rrf_top5": pool[:5], "llm_ranked_ids": valid_ranked_ids[:5],
                    "llm_top5": [id_to_spec[candidate_id] for candidate_id in valid_ranked_ids[:5]],
                    "parse_success": isinstance(parsed, dict), "parsed_output": parsed,
                    "error": error, "elapsed_seconds": time.perf_counter() - started,
                    "ollama_metadata": metadata,
                }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 10 == 0 or position == len(record_ids):
                print(f"{args.model} rerank seed={args.seed}: {position}/{len(record_ids)}", flush=True)


if __name__ == "__main__":
    main()
