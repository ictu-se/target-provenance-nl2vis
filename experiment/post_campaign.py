from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
CAMPAIGN_STATUS = HERE / "outputs" / "campaign_status.json"
STATUS = HERE / "outputs" / "post_campaign_status.json"
LOG = HERE / "outputs" / "post_campaign.log"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(payload: dict) -> None:
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def run_step(name: str, args: list[str], status: dict, log) -> None:
    step = next(item for item in status["steps"] if item["name"] == name)
    step.update({"status": "running", "started_utc": now()})
    write_status(status)
    command = [sys.executable, "-B", *args]
    log.write(f"\n[{now()}] START {name}\n{json.dumps(command, ensure_ascii=False)}\n")
    log.flush()
    result = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, text=True)
    step.update({"status": "completed" if result.returncode == 0 else "failed", "ended_utc": now(), "returncode": result.returncode})
    write_status(status)
    if result.returncode:
        raise RuntimeError(f"Step {name} failed with return code {result.returncode}")


def main() -> None:
    steps = [
        ("analyze_cross_family", ["analyze_forward.py", "--file-pattern", "*cross150.jsonl", "--output-dir", "out/a/cross"]),
        ("weight_cross_family", ["analyze_stratified_sample.py", "--per-case", "out/a/cross/per_case.csv", "--output", "out/a/cross/weighted.csv"]),
        ("paired_direct_staged", ["analyze_paired_conditions.py", "--per-case", "out/a/cross/per_case.csv", "--output", "out/a/cross/paired.csv"]),
        ("analyze_qwen_full", ["analyze_forward.py", "--file-pattern", "qwen3*_full751.jsonl", "--output-dir", "out/a/full"]),
        ("analyze_stability", ["analyze_stability.py", "--pattern", "qwen3*_repeat150.jsonl", "--output-dir", "out/a/stability"]),
        ("qwen_reranker", ["run_forward_reranker.py", "--model", "qwen3:14b", "--seed", "77"]),
        ("mistral_reranker", ["run_forward_reranker.py", "--model", "mistral-small:24b", "--seed", "77"]),
        ("analyze_rerankers", ["analyze_reranker.py", "--output-dir", "out/a/rerank"]),
        ("refresh_environment", ["capture_environment.py"]),
    ]
    status = {
        "created_utc": now(), "status": "waiting_for_campaign",
        "steps": [{"name": name, "status": "pending"} for name, _ in steps],
    }
    write_status(status)
    while True:
        if CAMPAIGN_STATUS.exists():
            campaign = json.loads(CAMPAIGN_STATUS.read_text(encoding="utf-8"))
            if campaign.get("status") == "completed":
                break
            if campaign.get("status") == "failed":
                status.update({"status": "blocked_by_failed_campaign", "ended_utc": now()})
                write_status(status)
                raise SystemExit(2)
        time.sleep(30)

    status["status"] = "running"
    write_status(status)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOG.open("a", encoding="utf-8") as log:
            for name, args in steps:
                run_step(name, args, status, log)
    except Exception as exc:
        status.update({"status": "failed", "ended_utc": now(), "error": f"{type(exc).__name__}: {exc}"})
        write_status(status)
        raise
    status.update({"status": "completed", "ended_utc": now()})
    write_status(status)


if __name__ == "__main__":
    main()
