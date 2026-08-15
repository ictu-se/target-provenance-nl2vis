from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    files = sorted(args.input_dir.glob("*.jsonl"))
    for path in files:
        for row in read_jsonl(path):
            grouped[(str(row["model"]), str(row["condition"]))].append(row)

    runs = []
    all_usable = True
    for (model, condition), rows in sorted(grouped.items()):
        parse_success = sum(bool(row.get("parse_success")) for row in rows)
        api_errors = sum(bool(row.get("error")) for row in rows)
        complete = len({row["record_id"] for row in rows}) == args.expected_cases
        usable = complete and parse_success > 0 and api_errors < len(rows)
        all_usable = all_usable and usable
        runs.append(
            {
                "model": model,
                "condition": condition,
                "rows": len(rows),
                "unique_cases": len({row["record_id"] for row in rows}),
                "parse_success_rows": parse_success,
                "api_error_rows": api_errors,
                "engineering_usable": usable,
            }
        )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision_basis": "completion, API errors, and parse success only; no gold, exact, graded, validity, or latency outcome was used for retention",
        "expected_runs": 12,
        "observed_runs": len(runs),
        "all_locked_runs_retained": all_usable and len(runs) == 12,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_locked_runs_retained"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
