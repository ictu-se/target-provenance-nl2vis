#!/usr/bin/env python3
"""Record the exact nvBench-v1 source used by the strict external adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.repo / "NVBench.json"
    current_bytes = source.read_bytes()
    committed_bytes = git(args.repo, "show", "HEAD:NVBench.json")
    current_payload = json.loads(current_bytes)
    committed_payload = json.loads(committed_bytes)
    if current_payload != committed_payload:
        raise SystemExit("Working NVBench.json is not semantically identical to the recorded commit")

    remote = git(args.repo, "remote", "get-url", "origin").decode().strip()
    commit = git(args.repo, "rev-parse", "HEAD").decode().strip()
    commit_time = git(args.repo, "show", "-s", "--format=%cI", "HEAD").decode().strip()
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": remote,
        "commit": commit,
        "commit_time": commit_time,
        "source_file": "NVBench.json",
        "records": len(current_payload),
        "working_sha256": hashlib.sha256(current_bytes).hexdigest(),
        "committed_sha256": hashlib.sha256(committed_bytes).hexdigest(),
        "byte_identical_to_commit": current_bytes == committed_bytes,
        "semantic_json_identical_to_commit": True,
        "byte_difference_interpretation": "line-ending normalization only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
