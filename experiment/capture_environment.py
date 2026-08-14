from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from common import sha256_file
from run_forward_ollama import DEFAULT_DATA, OLLAMA_URL


HERE = Path(__file__).resolve().parent


def command_output(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=30).json().get("models", [])
    scripts = sorted(HERE.glob("*.py"))
    payload = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "packages": {name: package_version(name) for name in ("requests", "scikit-learn", "numpy", "scipy", "matplotlib")},
        "ollama_version": command_output(["ollama", "--version"]),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "models": tags,
        "data_sha256": {split: sha256_file(DEFAULT_DATA / f"{split}.json") for split in ("train", "dev", "test")},
        "script_sha256": {path.name: sha256_file(path) for path in scripts},
    }
    output = HERE / "out" / "env.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "platform": payload["platform"], "gpu": payload["gpu"]}, indent=2))


if __name__ == "__main__":
    main()
