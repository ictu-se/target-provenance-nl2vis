"""Shared core-equivalence and exact-gain taxonomy helpers."""

from __future__ import annotations

import json
from typing import Any

from common import canonical_key, normalize_spec, spec_components, stable_json


AUXILIARY_ENCODING_KEYS = {"axis", "legend", "title", "format"}


def core_spec(spec: Any) -> dict[str, Any] | None:
    """Remove only the registered presentation metadata from a specification."""
    normalized = normalize_spec(spec)
    if normalized is None:
        return None
    output = json.loads(json.dumps(normalized))
    encoding = output.get("encoding", {})
    if isinstance(encoding, dict):
        for value in encoding.values():
            if isinstance(value, dict):
                for key in AUXILIARY_ENCODING_KEYS:
                    value.pop(key, None)
    for key in ("title", "description", "usermeta", "$schema", "config", "width", "height"):
        output.pop(key, None)
    return normalize_spec(output)


def core_key(spec: Any) -> str | None:
    core = core_spec(spec)
    return stable_json(core) if core is not None else None


def difference_taxonomy(pred: dict[str, Any] | None, gold: dict[str, Any]) -> str:
    """Classify the registered component differences behind an exact mismatch."""
    if pred is None:
        return "no_candidate"
    if core_key(pred) == core_key(gold) and canonical_key(pred) != canonical_key(gold):
        return "auxiliary_metadata_only"
    predicted, target = spec_components(pred), spec_components(gold)
    differences = [
        name
        for name in ("mark", "channels", "fields", "operations", "filters")
        if predicted[name] != target[name]
    ]
    if not differences:
        return "other_representation_difference"
    if differences == ["mark"]:
        return "mark_only"
    if differences == ["operations"]:
        return "operation_only"
    if differences == ["filters"]:
        return "filter_only"
    if set(differences).issubset({"channels", "fields"}):
        return "field_or_channel_mapping"
    return "multiple_components"
