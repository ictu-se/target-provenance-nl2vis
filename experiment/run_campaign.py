from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_forward_ollama.py"
SAMPLE = HERE / "design" / "forward_sample150.json"
OUTPUT = HERE / "out" / "f"
STATUS = HERE / "outputs" / "campaign_status.json"
LOG = HERE / "outputs" / "campaign.log"


PHASES = [
    {
        "name": "cross_family_direct_sample150",
        "args": [
            "--models", "qwen3:14b", "llama3.2:3b", "gemma3:27b", "mistral-small:24b",
            "--conditions", "direct", "--indices-file", str(SAMPLE), "--run-tag", "cross150",
            "--temperature", "0", "--seeds", "55",
        ],
    },
    {
        "name": "cross_family_staged_sample150",
        "args": [
            "--models", "qwen3:14b", "gemma3:27b", "mistral-small:24b",
            "--conditions", "staged", "--indices-file", str(SAMPLE), "--run-tag", "cross150",
            "--temperature", "0", "--seeds", "55",
        ],
    },
    {
        "name": "qwen_full_test",
        "args": [
            "--models", "qwen3:14b", "--conditions", "direct", "staged", "--run-tag", "full751",
            "--temperature", "0", "--seeds", "55",
        ],
    },
    {
        "name": "qwen_repeated_sample150",
        "args": [
            "--models", "qwen3:14b", "--conditions", "direct", "staged", "--indices-file", str(SAMPLE),
            "--run-tag", "repeat150", "--temperature", "0.2", "--seeds", "101", "202", "303",
        ],
    },
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(payload: dict) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def main() -> None:
    campaign = {
        "created_utc": now(),
        "status": "running",
        "runner": str(RUNNER),
        "sample": str(SAMPLE),
        "phases": [{"name": phase["name"], "status": "pending"} for phase in PHASES],
    }
    write_status(campaign)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        for index, phase in enumerate(PHASES):
            campaign["phases"][index].update({"status": "running", "started_utc": now()})
            write_status(campaign)
            command = [sys.executable, str(RUNNER), *phase["args"], "--output-dir", str(OUTPUT)]
            log.write(f"\n[{now()}] START {phase['name']}\n")
            log.write(json.dumps(command, ensure_ascii=False) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, text=True)
            campaign["phases"][index].update(
                {"status": "completed" if result.returncode == 0 else "failed", "ended_utc": now(), "returncode": result.returncode}
            )
            write_status(campaign)
            if result.returncode != 0:
                campaign.update({"status": "failed", "ended_utc": now(), "failed_phase": phase["name"]})
                write_status(campaign)
                raise SystemExit(result.returncode)
        campaign.update({"status": "completed", "ended_utc": now()})
        write_status(campaign)


if __name__ == "__main__":
    main()
