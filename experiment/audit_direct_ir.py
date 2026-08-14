from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import sha256_file


HERE = Path(__file__).resolve().parent
DEFAULT_LEGACY = Path(os.environ.get("INTENTLENS_LEGACY_DIR", HERE / "legacy_source"))


def strict_json_dict(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except Exception:
        return False


def inspect_condition(root: Path, name: str, train_summary_name: str) -> dict[str, Any]:
    condition_dir = root / "artifacts" / name
    predictions_path = condition_dir / "full_test_predictions.json"
    component_path = condition_dir / "full_test_component_summary.json"
    train_path = condition_dir / train_summary_name
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    components = json.loads(component_path.read_text(encoding="utf-8"))
    train_summary = json.loads(train_path.read_text(encoding="utf-8"))

    strict_parse_count = sum(strict_json_dict(str(row.get("prediction", ""))) for row in predictions)
    exact_count = sum(
        str(row.get("prediction", "")).strip() == str(row.get("gold", "")).strip()
        for row in predictions
    )
    ids = [str(row.get("record_id", "")) for row in predictions]
    malformed_examples = [
        {
            "record_id": row.get("record_id"),
            "prediction": row.get("prediction"),
            "gold": row.get("gold"),
        }
        for row in predictions
        if not strict_json_dict(str(row.get("prediction", "")))
    ][:3]

    return {
        "condition": name,
        "prediction_count": len(predictions),
        "unique_record_ids": len(set(ids)),
        "strict_json_parse_count": strict_parse_count,
        "strict_json_parse_rate": strict_parse_count / len(predictions) if predictions else 0.0,
        "raw_string_exact_count": exact_count,
        "raw_string_exact_rate": exact_count / len(predictions) if predictions else 0.0,
        "legacy_regex_parse_rate_claim": components["aggregate_metrics"].get("parseable_rate"),
        "legacy_slot_mean": components["aggregate_metrics"].get("slot_mean"),
        "legacy_component_metrics": components["aggregate_metrics"],
        "training_configuration": train_summary,
        "sha256": {
            "predictions": sha256_file(predictions_path),
            "component_summary": sha256_file(component_path),
            "training_summary": sha256_file(train_path),
        },
        "malformed_examples": malformed_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "direct_ir")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full = inspect_condition(args.legacy_dir, "train_long_t5_nvbench_full", "full_train_summary.json")
    lora = inspect_condition(args.legacy_dir, "train_long_t5_nvbench_lora", "adapter_train_summary.json")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_dir": str(args.legacy_dir.resolve()),
        "conditions": [full, lora],
        "audit_conclusion": {
            "strict_structured_generation_supported": False,
            "reason": "All 5,162 saved predictions fail strict JSON parsing. The legacy evaluator labels a row parseable when a regular expression finds a mark token, then extracts slots from malformed text. This does not support the manuscript claim of fully parseable IR.",
            "revision_action": "Remove the direct-IR branch from the ambiguity-ranking paper. Preserve it only in the audit record; do not cite slot_mean as a structured-generation baseline without a new strictly parsed rerun.",
            "rerun_needed_for_revised_scope": False,
            "rerun_condition": "Rerun only if direct IR remains a scientific contribution after scope revision; then require strict JSON/schema validation and multiple seeds.",
        },
    }
    output = args.output_dir / "audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["audit_conclusion"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
