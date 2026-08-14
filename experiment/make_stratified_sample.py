from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common import dedupe_specs, load_split, sha256_file


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
DEFAULT_DATA = Path(os.environ.get(
    "NVBENCH2_DATA_DIR",
    WORKSPACE / "data_benchmarks" / "datasets" / "nvBench-2.0" / "data" / "nvbench2.0",
))


def stratum(row: dict) -> str:
    golds = dedupe_specs(row["gold_answer"])
    return "+".join(sorted({str(gold.get("mark", "unknown")) for gold in golds})) or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", default="test")
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, default=HERE / "design" / "forward_sample150.json")
    args = parser.parse_args()

    rows = load_split(args.data_dir, args.split)
    if args.size <= 0 or args.size > len(rows):
        raise ValueError("size must be between 1 and the split size")
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(int(row["index"]))

    exact = {name: args.size * len(indices) / len(rows) for name, indices in groups.items()}
    if args.size < len(groups):
        raise ValueError("size must be at least the number of non-empty strata")
    allocation = {name: max(1, int(math.floor(value))) for name, value in exact.items()}
    remaining = args.size - sum(allocation.values())
    if remaining >= 0:
        for name in sorted(groups, key=lambda key: (-(exact[key] - allocation[key]), key))[:remaining]:
            allocation[name] += 1
    else:
        removable = sorted(groups, key=lambda key: (exact[key] - allocation[key], key))
        for name in removable:
            if remaining == 0:
                break
            if allocation[name] > 1:
                allocation[name] -= 1
                remaining += 1

    rng = random.Random(args.seed)
    selected: list[int] = []
    per_stratum: dict[str, dict[str, int]] = {}
    for name in sorted(groups):
        population = sorted(groups[name])
        sample = sorted(rng.sample(population, allocation[name]))
        selected.extend(sample)
        per_stratum[name] = {"population": len(population), "sample": len(sample)}

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": str((args.data_dir / f"{args.split}.json").resolve()),
        "data_sha256": sha256_file(args.data_dir / f"{args.split}.json"),
        "split": args.split,
        "sampling_seed": args.seed,
        "sampling_policy": "proportional allocation by canonical gold mark-family stratum using largest remainders; simple random sample without replacement within strata",
        "population_size": len(rows),
        "sample_size": len(selected),
        "strata": per_stratum,
        "indices": sorted(selected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("population_size", "sample_size", "strata")}, indent=2))


if __name__ == "__main__":
    main()
