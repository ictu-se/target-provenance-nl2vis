#!/usr/bin/env python3
"""Create, update, upload, and publish this manuscript-free Zenodo record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import requests


DEFAULT_API = "https://zenodo.org/api"
TITLE = "Target-Provenance Audits for Ambiguous NL2Vis: Code and Experimental Results"


def env_value(path: Path, key: str) -> str:
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'\"")
    raise KeyError(f"{key} not found in {path}")


def check(response: requests.Response, expected: set[int]) -> dict[str, Any]:
    if response.status_code not in expected:
        body = response.text[:1000]
        raise RuntimeError(f"Zenodo HTTP {response.status_code}: {body}")
    if not response.content:
        return {}
    return response.json()


def headers(token: str, json_content: bool = False) -> dict[str, str]:
    value = {"Authorization": f"Bearer {token}"}
    if json_content:
        value["Content-Type"] = "application/json"
    return value


def metadata(version: str) -> dict[str, Any]:
    return {
        "title": TITLE,
        "upload_type": "software",
        "description": (
            "Code, frozen experimental designs, prompts, benchmark-derived inputs, "
            "raw local-model outputs, provenance audits, rendered experimental figures, "
            "and statistical result tables for the accompanying target-provenance study. "
            "The archive excludes the manuscript and submission documents."
        ),
        "creators": [
            {
                "name": "Nguyen The-Vinh",
                "affiliation": "Thai Nguyen University of Information and Communication Technology",
            }
        ],
        "access_right": "open",
        "license": "MIT",
        "version": version,
        "language": "eng",
        "keywords": [
            "natural language to visualization",
            "NL2Vis",
            "target provenance",
            "benchmark leakage",
            "ranked visualization alternatives",
            "evaluation audit",
        ],
        "prereserve_doi": True,
    }


def find_or_create_draft(api: str, token: str, version: str) -> dict[str, Any]:
    response = requests.get(
        f"{api}/deposit/depositions",
        headers=headers(token),
        params={"status": "draft", "sort": "mostrecent", "size": 100},
        timeout=60,
    )
    drafts = check(response, {200})
    for draft in drafts:
        if draft.get("metadata", {}).get("title") == TITLE:
            return draft
    response = requests.post(
        f"{api}/deposit/depositions",
        headers=headers(token, json_content=True),
        json={"metadata": metadata(version)},
        timeout=60,
    )
    return check(response, {201})


def new_version_draft(api: str, token: str, deposition_id: int) -> dict[str, Any]:
    response = requests.post(
        f"{api}/deposit/depositions/{deposition_id}/actions/newversion",
        headers=headers(token),
        timeout=120,
    )
    original = check(response, {201})
    latest_draft = original.get("links", {}).get("latest_draft")
    if not latest_draft:
        raise RuntimeError("Zenodo new-version response has no links.latest_draft")
    return check(requests.get(latest_draft, headers=headers(token), timeout=60), {200})


def state_draft(
    api: str, token: str, state_file: Optional[Path]
) -> Optional[dict[str, Any]]:
    """Reuse a previously reserved draft so reruns do not request another version."""
    if state_file is None or not state_file.exists():
        return None
    state = json.loads(state_file.read_text(encoding="utf-8"))
    deposition_id = state.get("id")
    if not deposition_id:
        return None
    response = requests.get(
        f"{api}/deposit/depositions/{int(deposition_id)}",
        headers=headers(token),
        timeout=60,
    )
    if response.status_code == 404:
        return None
    draft = check(response, {200})
    if draft.get("submitted"):
        return None
    if draft.get("metadata", {}).get("title") != TITLE:
        raise RuntimeError(
            f"State file points to a different Zenodo deposition: {deposition_id}"
        )
    return draft


def update_metadata(
    api: str, token: str, deposition_id: int, version: str
) -> dict[str, Any]:
    response = requests.put(
        f"{api}/deposit/depositions/{deposition_id}",
        headers=headers(token, json_content=True),
        json={"metadata": metadata(version)},
        timeout=60,
    )
    return check(response, {200})


def delete_inherited_files(draft: dict[str, Any], token: str) -> None:
    files_url = draft.get("links", {}).get("files")
    if not files_url:
        return
    files = check(requests.get(files_url, headers=headers(token), timeout=60), {200})
    for item in files:
        delete_url = item.get("links", {}).get("self")
        if not delete_url and item.get("id"):
            delete_url = f"{files_url.rstrip('/')}/{item['id']}"
        if not delete_url:
            continue
        response = requests.delete(delete_url, headers=headers(token), timeout=60)
        check(response, {204})


def upload_archive(draft: dict[str, Any], token: str, archive: Path) -> dict[str, Any]:
    bucket = draft["links"]["bucket"].rstrip("/")
    response = requests.put(
        f"{bucket}/{archive.name}",
        headers=headers(token),
        data=archive.open("rb"),
        timeout=1800,
    )
    return check(response, {200, 201})


def publish(api: str, token: str, deposition_id: int) -> dict[str, Any]:
    response = requests.post(
        f"{api}/deposit/depositions/{deposition_id}/actions/publish",
        headers=headers(token),
        timeout=120,
    )
    return check(response, {202})


def public_summary(record: dict[str, Any]) -> dict[str, Any]:
    reserved = record.get("metadata", {}).get("prereserve_doi", {})
    doi = record.get("doi") or reserved.get("doi")
    return {
        "id": record.get("id"),
        "state": record.get("state"),
        "submitted": record.get("submitted"),
        "doi": doi,
        "doi_url": record.get("doi_url") or (f"https://doi.org/{doi}" if doi else None),
        "record_url": record.get("record_url"),
        "bucket": record.get("links", {}).get("bucket"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--new-version-of", type=int)
    parser.add_argument("--version", default="1.1.1")
    args = parser.parse_args()

    token = env_value(args.env_file, "ZENODO_ACCESS_TOKEN")
    api = args.api.rstrip("/")
    if args.new_version_of:
        draft = state_draft(api, token, args.state_file)
        if draft is None:
            draft = new_version_draft(api, token, args.new_version_of)
    else:
        draft = find_or_create_draft(api, token, args.version)
    draft = update_metadata(api, token, int(draft["id"]), args.version)
    if args.archive:
        if args.new_version_of:
            delete_inherited_files(draft, token)
        upload_archive(draft, token, args.archive)
        draft = check(
            requests.get(draft["links"]["self"], headers=headers(token), timeout=60),
            {200},
        )
    if args.publish:
        if not args.archive:
            raise ValueError("--publish requires --archive")
        draft = publish(api, token, int(draft["id"]))

    summary = public_summary(draft)
    if args.state_file:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
