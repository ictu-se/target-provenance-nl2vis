from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "reviewer_revision_20260807"
if not BASE.exists():
    BASE = HERE.parent
sys.path.insert(0, str(BASE))

from common import dedupe_specs, load_split, sha256_file  # noqa: E402
from run_forward_ollama import (  # noqa: E402
    CHART_RULES,
    DIRECT_RICH_TEMPLATE,
    DIRECT_TEMPLATE,
    STAGED_TEMPLATE,
)


def stratum(row: dict) -> str:
    golds = dedupe_specs(row["gold_answer"])
    return "+".join(sorted({str(gold.get("mark", "unknown")) for gold in golds})) or "unknown"


def allocate(groups: dict[str, list[int]], size: int) -> dict[str, int]:
    total = sum(len(indices) for indices in groups.values())
    exact = {name: size * len(indices) / total for name, indices in groups.items()}
    allocation = {name: max(1, math.floor(value)) for name, value in exact.items()}
    while sum(allocation.values()) < size:
        name = max(groups, key=lambda key: (exact[key] - allocation[key], key))
        allocation[name] += 1
    while sum(allocation.values()) > size:
        candidates = [name for name in groups if allocation[name] > 1]
        name = min(candidates, key=lambda key: (exact[key] - allocation[key], key))
        allocation[name] -= 1
    return allocation


def sample_stratified(
    groups: dict[str, list[int]], size: int, rng: random.Random
) -> tuple[list[int], dict[str, int]]:
    allocation = allocate(groups, size)
    selected: list[int] = []
    for name in sorted(groups):
        selected.extend(rng.sample(sorted(groups[name]), allocation[name]))
    return sorted(selected), allocation


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--screen-size", type=int, default=30)
    parser.add_argument("--holdout-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output-dir", type=Path, default=HERE / "design")
    args = parser.parse_args()

    rows = load_split(args.data_dir, "dev")
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(int(row["index"]))

    rng = random.Random(args.seed)
    screen, screen_allocation = sample_stratified(groups, args.screen_size, rng)
    screen_set = set(screen)
    remaining = {
        name: [index for index in indices if index not in screen_set]
        for name, indices in groups.items()
    }
    holdout, holdout_allocation = sample_stratified(remaining, args.holdout_size, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": str((args.data_dir / "dev.json").resolve()),
        "data_sha256": sha256_file(args.data_dir / "dev.json"),
        "split": "dev",
        "sampling_seed": args.seed,
        "stratum": "set of canonical gold mark families",
    }
    screen_payload = {
        **common,
        "role": "engineering screen only; no model, prompt, or condition may be removed using outcome quality",
        "population_size": len(rows),
        "sample_size": len(screen),
        "allocation": screen_allocation,
        "indices": screen,
    }
    holdout_payload = {
        **common,
        "role": "locked post-revision holdout; disjoint from engineering screen and historical test runs",
        "population_size_after_screen_exclusion": len(rows) - len(screen),
        "sample_size": len(holdout),
        "population_strata_after_screen": {name: len(indices) for name, indices in remaining.items()},
        "allocation": holdout_allocation,
        "indices": holdout,
    }
    protocol = {
        "status": "locked_before_round2_inference",
        "created_utc": common["created_utc"],
        "models": ["gemma3:27b", "llama3.2:3b", "mistral-small:24b", "qwen3:14b"],
        "conditions": ["direct", "direct_rich", "staged"],
        "primary_contrasts": ["direct_rich-minus-direct", "staged-minus-direct_rich"],
        "secondary_contrast": "staged-minus-direct",
        "generation": {"temperature": 0.0, "seed": 55, "num_ctx": 8192, "num_predict": 3072},
        "screen_decision_rule": "retain all locked models and conditions unless an API/model failure prevents JSON generation; outcome metrics cannot exclude a run",
        "holdout_rule": "never use the 150 holdout cases for model or prompt selection",
        "prompt_sha256": {
            "chart_rules": digest(CHART_RULES),
            "direct": digest(DIRECT_TEMPLATE),
            "direct_rich": digest(DIRECT_RICH_TEMPLATE),
            "staged": digest(STAGED_TEMPLATE),
        },
        "screen_design_sha256": "filled_after_write",
        "holdout_design_sha256": "filled_after_write",
    }
    screen_path = args.output_dir / "dev_engineering_screen30.json"
    holdout_path = args.output_dir / "dev_locked_holdout150.json"
    screen_path.write_text(json.dumps(screen_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    holdout_path.write_text(json.dumps(holdout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol["screen_design_sha256"] = sha256_file(screen_path)
    protocol["holdout_design_sha256"] = sha256_file(holdout_path)
    (args.output_dir / "locked_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"screen": screen_payload, "holdout": holdout_payload, "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
