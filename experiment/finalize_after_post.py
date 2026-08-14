from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
POST_STATUS = HERE / "outputs" / "post_campaign_status.json"
STATUS = HERE / "outputs" / "finalization_status.json"
LOG = HERE / "outputs" / "finalization.log"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(payload: dict) -> None:
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def main() -> None:
    status = {"created_utc": now(), "status": "waiting_for_post_campaign"}
    write(status)
    while True:
        if POST_STATUS.exists():
            post = json.loads(POST_STATUS.read_text(encoding="utf-8"))
            if post.get("status") == "completed":
                break
            if post.get("status") in {"failed", "blocked_by_failed_campaign"}:
                status.update({"status": "blocked", "ended_utc": now(), "post_status": post.get("status")})
                write(status)
                raise SystemExit(2)
        time.sleep(30)

    status["status"] = "running"
    write(status)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        for script in ("validate_outputs.py", "make_final_outputs.py"):
            result = subprocess.run([sys.executable, "-B", script], cwd=HERE, stdout=log, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                status.update({"status": "failed", "failed_script": script, "ended_utc": now(), "returncode": result.returncode})
                write(status)
                raise SystemExit(result.returncode)
            if script == "validate_outputs.py":
                validation = json.loads((HERE / "out" / "validation.json").read_text(encoding="utf-8"))
                if validation.get("status") != "complete":
                    status.update({"status": "failed_integrity", "ended_utc": now(), "validation": validation})
                    write(status)
                    raise SystemExit(3)
    status.update({"status": "completed", "ended_utc": now(), "canonical_results": str((HERE / "out" / "final" / "canonical_results.json").resolve())})
    write(status)


if __name__ == "__main__":
    main()
