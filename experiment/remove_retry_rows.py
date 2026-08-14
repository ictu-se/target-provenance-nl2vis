from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove selected JSONL records for a resumable retry.")
    parser.add_argument("path", type=Path)
    parser.add_argument("record_ids", nargs="+")
    args = parser.parse_args()

    requested = set(args.record_ids)
    source_lines = args.path.read_text(encoding="utf-8").splitlines()
    retained: list[str] = []
    removed: list[dict] = []
    for line_number, line in enumerate(source_lines, start=1):
        payload = json.loads(line)
        if str(payload.get("record_id")) in requested:
            removed.append({"line": line_number, "record_id": payload.get("record_id"), "error": payload.get("error")})
        else:
            retained.append(json.dumps(payload, ensure_ascii=False))
    found = {str(row["record_id"]) for row in removed}
    if found != requested:
        raise ValueError(f"Requested IDs not found exactly; requested={sorted(requested)}, found={sorted(found)}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_id = hashlib.sha256(str(args.path.resolve()).encode("utf-8")).hexdigest()[:8]
    backup = args.path.parent / f"retry_b_{short_id}_{timestamp}.jsonl"
    audit = args.path.parent / f"retry_a_{short_id}_{timestamp}.json"
    shutil.copy2(args.path, backup)
    args.path.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    audit.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "path": str(args.path.resolve()),
        "backup": str(backup.resolve()), "input_rows": len(source_lines), "retained_rows": len(retained),
        "removed": removed, "purpose": "resumable retry of explicitly selected records",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(args.path), "removed": removed, "backup": str(backup)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
