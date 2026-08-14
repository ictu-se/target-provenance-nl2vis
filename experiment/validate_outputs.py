from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
VERIFIED_FAILURES = HERE / "design" / "verified_inference_failures.json"


def check_jsonl(path: Path, expected: int | None, verified: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                errors.append(f"line {line_number}: {type(exc).__name__}: {exc}")
    ids = [str(row.get("record_id")) for row in rows]
    duplicate_count = len(ids) - len(set(ids))
    allowed_outcome_errors = {"empty_valid_candidate_pool"}
    outcome_error_rows = sum(row.get("error") in allowed_outcome_errors for row in rows)
    verified_failure_rows = 0
    transport_error_rows = 0
    for row in rows:
        error = str(row.get("error") or "")
        if not error or error in allowed_outcome_errors:
            continue
        matched = any(
            item.get("file") == path.name
            and item.get("record_id") == row.get("record_id")
            and str(item.get("error_contains", "")) in error
            for item in verified
        )
        if matched:
            verified_failure_rows += 1
        else:
            transport_error_rows += 1
    return {
        "file": str(path.resolve()), "rows": len(rows), "unique_record_ids": len(set(ids)),
        "duplicate_record_ids": duplicate_count, "parse_errors": errors, "expected_rows": expected,
        "complete": expected is None or (len(rows) == expected and duplicate_count == 0 and not errors and transport_error_rows == 0),
        "transport_error_rows": transport_error_rows,
        "outcome_error_rows": outcome_error_rows,
        "verified_inference_failure_rows": verified_failure_rows,
    }


def expected_forward(path: Path) -> int | None:
    name = path.name
    if "cross150" in name or "repeat150" in name:
        return 150
    if "full751" in name:
        return 751
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "out" / "validation.json")
    args = parser.parse_args()

    verified = json.loads(VERIFIED_FAILURES.read_text(encoding="utf-8")).get("failures", [])
    forward = [
        check_jsonl(path, expected_forward(path), verified)
        for path in sorted((HERE / "out" / "f").glob("*.jsonl"))
        if not path.name.startswith("b_")
    ]
    rerank = [check_jsonl(path, 150, verified) for path in sorted((HERE / "out" / "r").glob("*.jsonl"))]
    required = [row for row in forward if row["expected_rows"] is not None] + rerank
    payload = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if required and all(row["complete"] for row in required) else "incomplete",
        "forward": forward, "reranker": rerank,
        "verified_failure_registry": str(VERIFIED_FAILURES.resolve()),
        "required_file_count": len(required),
        "complete_required_file_count": sum(row["complete"] for row in required),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "required_file_count", "complete_required_file_count")}, indent=2))


if __name__ == "__main__":
    main()
