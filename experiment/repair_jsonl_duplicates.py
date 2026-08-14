from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair duplicate JSONL records while preserving an audit copy.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--key", default="record_id")
    args = parser.parse_args()

    source_lines = args.path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    retained: list[str] = []
    duplicates: list[dict[str, object]] = []
    for line_number, line in enumerate(source_lines, start=1):
        payload = json.loads(line)
        value = str(payload[args.key])
        if value in seen:
            duplicates.append({"line": line_number, args.key: value})
            continue
        seen.add(value)
        retained.append(json.dumps(payload, ensure_ascii=False))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_id = hashlib.sha256(str(args.path.resolve()).encode("utf-8")).hexdigest()[:8]
    backup = args.path.parent / f"b_{short_id}_{timestamp}.jsonl"
    shutil.copy2(args.path, backup)
    args.path.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    audit = {
        "repaired_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(args.path.resolve()),
        "backup": str(backup.resolve()),
        "key": args.key,
        "input_lines": len(source_lines),
        "retained_lines": len(retained),
        "duplicates_removed": duplicates,
        "policy": "retain first physical occurrence for each key",
        "cause": "two resumable workers briefly wrote the same run after an outer shell timeout left one worker alive",
    }
    audit_path = args.path.parent / f"a_{short_id}_{timestamp}.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
