#!/usr/bin/env python3
"""Rerank all-TAF executable-and-normal-form-compliant candidate pools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
sys.path.insert(0, str(BASE))

from common import canonical_key, extract_first_json, load_split, sha256_file  # noqa: E402
from run_forward_ollama import OLLAMA_URL, compact_model_slug, get_model_manifest  # noqa: E402
from run_forward_reranker import make_prompt  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed(path: Path) -> set[str]:
    return {str(row["record_id"]) for row in read_jsonl(path)} if path.exists() else set()


def renderability_map(path: Path) -> dict[tuple[str, str, str, int], bool]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row["record_id"], row["model"], row["condition"], int(row["rank"])): bool(int(row["renderer_executable"]))
            for row in csv.DictReader(handle)
        }


def load_sources(input_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paths = sorted(input_dir.glob("*.jsonl"))
    for path in paths:
        for row in read_jsonl(path):
            grouped[str(row["record_id"])].append(row)
    return grouped, paths


def build_eligible_pool(
    rows: list[dict[str, Any]], executable: dict[tuple[str, str, str, int], bool], excluded_model: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    by_key: dict[str, dict[str, Any]] = {}
    eligible_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_keys: set[str] = set()
    for row in rows:
        model, condition, record_id = str(row["model"]), str(row["condition"]), str(row["record_id"])
        validation = row.get("candidate_validation_errors", [])
        for rank, candidate in enumerate(row.get("candidates", []), start=1):
            key = canonical_key(candidate)
            if key is None:
                continue
            raw_keys.add(key)
            compliant = rank - 1 < len(validation) and not validation[rank - 1]
            renderer_ok = executable.get((record_id, model, condition, rank), False)
            if not compliant or not renderer_ok or (excluded_model is not None and model == excluded_model):
                continue
            by_key.setdefault(key, candidate)
            eligible_sources[key].append({
                "model": model, "condition": condition, "source_run": f"{model}::{condition}", "rank": rank,
            })
    scored = []
    for key, candidate in by_key.items():
        score = sum(1.0 / (60.0 + source["rank"]) for source in eligible_sources[key])
        scored.append((score, key, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return (
        [candidate for _, _, candidate in scored],
        [{"canonical_key": key, "rrf_score": score, "sources": eligible_sources[key]} for score, key, _ in scored],
        len(raw_keys),
    )


def exact_length_call(model: str, prompt: str, ids: list[str], seed: int, timeout: int) -> tuple[Any, dict[str, Any]]:
    target = min(5, len(ids))
    schema = {
        "type": "object", "required": ["ranked_ids"], "additionalProperties": False,
        "properties": {"ranked_ids": {
            "type": "array", "minItems": target, "maxItems": target, "uniqueItems": True,
            "items": {"type": "string", "enum": ids},
        }},
    }
    payload = {
        "model": model, "prompt": prompt, "stream": False, "format": schema, "think": False,
        "options": {"temperature": 0.0, "seed": seed, "num_predict": 256, "num_ctx": 8192},
        "keep_alive": "15m",
    }
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            parsed = extract_first_json(str(body.get("response", "")))
            ranked = parsed.get("ranked_ids", []) if isinstance(parsed, dict) else []
            if len(ranked) != target:
                raise ValueError(f"expected {target} ranked IDs, received {len(ranked)}")
            metadata = {key: body.get(key) for key in (
                "done_reason", "total_duration", "load_duration", "prompt_eval_count", "eval_count", "eval_duration"
            )}
            return parsed, metadata
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"exact-length reranker failed after 3 attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--renderability", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", choices=("full", "leave_self_out"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    sources, source_paths = load_sources(args.input_dir)
    executable = renderability_map(args.renderability)
    gold = {row["record_id"]: row for row in load_split(args.data_dir, "dev")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = f"all_taf_eligible_{args.variant}"
    output = args.output_dir / f"{compact_model_slug(args.model)}_{args.seed}_{label}.jsonl"
    done = completed(output)
    manifest = {
        "status": "running", "model": args.model, "model_manifest": get_model_manifest().get(args.model, {}),
        "variant": args.variant, "seed": args.seed, "temperature": 0.0,
        "source_files": [path.name for path in source_paths],
        "source_sha256": {path.name: sha256_file(path) for path in source_paths},
        "conditions": ["direct", "direct_rich", "staged"],
        "eligibility": "renderer executable AND study-registered benchmark-normal-form compliant",
        "excluded_model": args.model if args.variant == "leave_self_out" else None,
        "baseline": "RRF k0=60 over eligible source ranks with canonical-key tie breaking",
        "output_policy": "exactly min(5,pool_size) distinct existing IDs; no backfill",
    }
    manifest_path = args.output_dir / f"manifest_{compact_model_slug(args.model)}_{args.seed}_{label}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    record_ids = sorted(sources, key=lambda value: int(value.rsplit(":", 1)[-1]))
    if args.limit is not None:
        record_ids = record_ids[: args.limit]
    with output.open("a", encoding="utf-8") as handle:
        for position, record_id in enumerate(record_ids, start=1):
            if record_id in done:
                continue
            excluded = args.model if args.variant == "leave_self_out" else None
            pool, details, raw_count = build_eligible_pool(sources[record_id], executable, excluded)
            row = gold[record_id]
            if not pool:
                result = {
                    "record_id": record_id, "index": row["index"], "model": args.model, "variant": args.variant,
                    "pool_name": label, "seed": args.seed, "temperature": 0.0, "raw_union_count": raw_count,
                    "pool": [], "pool_details": [], "rrf_top5": [], "llm_ranked_ids": [], "llm_top5": [],
                    "parse_success": False, "prefix_complete": True, "error": "empty_eligible_pool", "elapsed_seconds": 0.0,
                }
            else:
                pairs = [(f"c{index}", spec) for index, spec in enumerate(pool)]
                presented = list(pairs)
                random.Random(f"{record_id}:{args.seed}:{args.variant}:{args.model}").shuffle(presented)
                prompt = make_prompt(row, presented).replace(
                    "Rank up to five distinct candidates", f"Rank exactly {min(5, len(pool))} distinct candidates"
                )
                started = time.perf_counter()
                error = None
                parsed: Any = None
                metadata: dict[str, Any] = {}
                try:
                    parsed, metadata = exact_length_call(
                        args.model, prompt, [candidate_id for candidate_id, _ in presented], args.seed, args.timeout
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                ranked = parsed.get("ranked_ids", []) if isinstance(parsed, dict) else []
                id_to_spec = dict(pairs)
                ranked = [candidate_id for candidate_id in ranked if candidate_id in id_to_spec]
                target = min(5, len(pool))
                result = {
                    "record_id": record_id, "index": row["index"], "model": args.model, "variant": args.variant,
                    "pool_name": label, "seed": args.seed, "temperature": 0.0,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "raw_union_count": raw_count, "eligible_pool_count": len(pool),
                    "pool": pool, "pool_details": details,
                    "presentation_order": [candidate_id for candidate_id, _ in presented],
                    "rrf_top5": pool[:5], "llm_ranked_ids": ranked[:5],
                    "llm_top5": [id_to_spec[candidate_id] for candidate_id in ranked[:5]],
                    "parse_success": isinstance(parsed, dict), "prefix_complete": len(ranked) == target,
                    "parsed_output": parsed, "error": error, "elapsed_seconds": time.perf_counter() - started,
                    "ollama_metadata": metadata,
                }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 10 == 0 or position == len(record_ids):
                print(f"{args.model} {args.variant}: {position}/{len(record_ids)}", flush=True)
    manifest["status"] = "complete" if len(completed(output)) == len(record_ids) else "incomplete"
    manifest["rows"] = len(completed(output))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
