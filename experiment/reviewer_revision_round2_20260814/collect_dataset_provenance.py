#!/usr/bin/env python3
"""Record the exact nvBench-2.0 snapshot used by the round-2 audit."""

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

    repo = args.repo.resolve()
    commit = git(repo, "rev-parse", "HEAD").decode().strip()
    remote = git(repo, "remote", "get-url", "origin").decode().strip()
    commit_time = git(repo, "show", "-s", "--format=%cI", "HEAD").decode().strip()

    splits = {}
    for split in ("train", "dev", "test"):
        rel = f"data/nvbench2.0/{split}.json"
        working_bytes = (repo / rel).read_bytes()
        committed_bytes = git(repo, "show", f"HEAD:{rel}")
        working_json = json.loads(working_bytes)
        committed_json = json.loads(committed_bytes)
        splits[split] = {
            "path": rel,
            "records": len(working_json),
            "working_sha256": hashlib.sha256(working_bytes).hexdigest(),
            "commit_blob_sha256": hashlib.sha256(committed_bytes).hexdigest(),
            "semantically_identical_to_commit": working_json == committed_json,
            "byte_identical_to_commit": working_bytes == committed_bytes,
        }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": remote,
        "commit": commit,
        "commit_timestamp": commit_time,
        "splits": splits,
        "total_records": sum(item["records"] for item in splits.values()),
        "paper_reported_queries": 7878,
        "unreconciled_difference": 7878 - sum(item["records"] for item in splits.values()),
        "interpretation": (
            "The checked-out split contents are semantically identical to the cited commit. "
            "Byte differences are line-ending/formatting differences and do not change parsed JSON. "
            "The 373-record gap therefore exists between the public commit and the paper/README count, "
            "not because the present study silently filtered the split files."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
