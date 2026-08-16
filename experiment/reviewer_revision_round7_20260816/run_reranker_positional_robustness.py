#!/usr/bin/env python3
"""Rerun the full eligible union under locked candidate permutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROUND5 = HERE.parent / "reviewer_revision_round5_20260815"
BASE = HERE.parent / "reviewer_revision_20260807"
if not (BASE / "common.py").exists():
    BASE = HERE.parent
sys.path.insert(0, str(ROUND5))
sys.path.insert(0, str(BASE))

from common import load_split  # noqa: E402
from run_eligible_reranker import (  # noqa: E402
    build_eligible_pool,
    completed,
    exact_length_call,
    load_sources,
    renderability_map,
)
from run_forward_ollama import compact_model_slug, get_model_manifest  # noqa: E402
from run_forward_reranker import make_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--renderability", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--permutation-seed", type=int, required=True)
    parser.add_argument("--decoding-seed", type=int, default=91)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    sources, source_paths = load_sources(args.input_dir)
    executable = renderability_map(args.renderability)
    gold = {row["record_id"]: row for row in load_split(args.data_dir, "dev")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = compact_model_slug(args.model)
    output = args.output_dir / f"{slug}_decode{args.decoding_seed}_perm{args.permutation_seed}_full.jsonl"
    done = completed(output)
    manifest_path = args.output_dir / f"manifest_{slug}_decode{args.decoding_seed}_perm{args.permutation_seed}_full.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "model": args.model,
        "model_manifest": get_model_manifest().get(args.model, {}),
        "variant": "full",
        "decoding_seed": args.decoding_seed,
        "permutation_seed": args.permutation_seed,
        "temperature": 0.0,
        "source_files": [path.name for path in source_paths],
        "eligibility": "standardized-completion executable AND study-registered benchmark-normal-form compliant",
        "baseline": "RRF k0=60 over eligible source ranks with canonical-key tie breaking",
        "output_policy": "exactly min(5,pool_size) distinct existing IDs; no backfill",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    record_ids = sorted(sources, key=lambda value: int(value.rsplit(":", 1)[-1]))
    if args.limit is not None:
        record_ids = record_ids[: args.limit]

    with output.open("a", encoding="utf-8") as handle:
        for position, record_id in enumerate(record_ids, start=1):
            if record_id in done:
                continue
            pool, details, raw_count = build_eligible_pool(sources[record_id], executable, None)
            row = gold[record_id]
            if not pool:
                result = {
                    "record_id": record_id,
                    "index": row["index"],
                    "model": args.model,
                    "variant": "full",
                    "decoding_seed": args.decoding_seed,
                    "permutation_seed": args.permutation_seed,
                    "temperature": 0.0,
                    "raw_union_count": raw_count,
                    "pool": [],
                    "pool_details": [],
                    "presentation_order": [],
                    "rrf_top5": [],
                    "llm_ranked_ids": [],
                    "llm_top5": [],
                    "parse_success": False,
                    "prefix_complete": True,
                    "error": "empty_eligible_pool",
                    "elapsed_seconds": 0.0,
                }
            else:
                pairs = [(f"c{index}", spec) for index, spec in enumerate(pool)]
                presented = list(pairs)
                random.Random(f"{record_id}:{args.permutation_seed}:full:{args.model}").shuffle(presented)
                prompt = make_prompt(row, presented).replace(
                    "Rank up to five distinct candidates",
                    f"Rank exactly {min(5, len(pool))} distinct candidates",
                )
                started = time.perf_counter()
                parsed: Any = None
                metadata: dict[str, Any] = {}
                error = None
                try:
                    parsed, metadata = exact_length_call(
                        args.model,
                        prompt,
                        [candidate_id for candidate_id, _ in presented],
                        args.decoding_seed,
                        args.timeout,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                ranked = parsed.get("ranked_ids", []) if isinstance(parsed, dict) else []
                id_to_spec = dict(pairs)
                ranked = [candidate_id for candidate_id in ranked if candidate_id in id_to_spec]
                target = min(5, len(pool))
                result = {
                    "record_id": record_id,
                    "index": row["index"],
                    "model": args.model,
                    "variant": "full",
                    "decoding_seed": args.decoding_seed,
                    "permutation_seed": args.permutation_seed,
                    "temperature": 0.0,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "raw_union_count": raw_count,
                    "eligible_pool_count": len(pool),
                    "pool": pool,
                    "pool_details": details,
                    "presentation_order": [candidate_id for candidate_id, _ in presented],
                    "rrf_top5": pool[:5],
                    "llm_ranked_ids": ranked[:5],
                    "llm_top5": [id_to_spec[candidate_id] for candidate_id in ranked[:5]],
                    "parse_success": isinstance(parsed, dict),
                    "prefix_complete": len(ranked) == target,
                    "parsed_output": parsed,
                    "error": error,
                    "elapsed_seconds": time.perf_counter() - started,
                    "ollama_metadata": metadata,
                }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 10 == 0 or position == len(record_ids):
                print(f"{args.model} permutation {args.permutation_seed}: {position}/{len(record_ids)}", flush=True)

    manifest["status"] = "complete" if len(completed(output)) == len(record_ids) else "incomplete"
    manifest["rows"] = len(completed(output))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
