from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable


MARK_ALIASES = {
    "pie": "arc",
    "piechart": "arc",
    "barchart": "bar",
    "linechart": "line",
    "scatter": "point",
    "scatterplot": "point",
    "heatmap": "rect",
    "heat-map": "rect",
    "box": "boxplot",
    "boxplot": "boxplot",
}
ALLOWED_MARKS = {"bar", "line", "arc", "point", "rect", "boxplot"}
CHANNELS = ("x", "y", "theta", "color", "size", "row", "column", "detail")
OPERATIONS = ("aggregate", "bin", "timeUnit", "sort")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_jsonish(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return deepcopy(fallback)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return deepcopy(fallback)


def load_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = data_dir / f"{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        csv_file = str(raw.get("csv_file", ""))
        rows.append(
            {
                "split": split,
                "index": index,
                "record_id": str(raw.get("record_id", f"nvbench2:{split}:{index}")),
                "csv_file": csv_file,
                "nl_query": " ".join(str(raw.get("nl_query", "")).split()),
                "table_schema": parse_jsonish(raw.get("table_schema"), {}),
                "steps": parse_jsonish(raw.get("steps"), {}),
                "gold_answer": parse_jsonish(raw.get("gold_answer"), []),
                "source_benchmark": str(raw.get("source_benchmark", "nvBench-2.0")),
                "chart_family": str(raw.get("chart_family", "")),
                "db_id": str(raw.get("db_id", "")),
            }
        )
    return rows


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _normalize_nested(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_nested(child, str(key))
            for key, child in sorted(value.items(), key=lambda pair: str(pair[0]))
            if child is not None and child != "" and child != [] and child != {}
        }
    if isinstance(value, list):
        normalized = [_normalize_nested(child, parent_key) for child in value]
        if parent_key in {"oneOf", "transform"} or all(isinstance(child, dict) for child in normalized):
            return sorted(normalized, key=stable_json)
        return normalized
    return _normalize_scalar(value)


def normalize_spec(spec: Any) -> dict[str, Any] | None:
    if not isinstance(spec, dict):
        return None
    candidate = deepcopy(spec)
    raw_mark = candidate.get("mark")
    if isinstance(raw_mark, dict):
        raw_mark = raw_mark.get("type")
    if isinstance(raw_mark, str):
        mark_key = re.sub(r"[ _]", "", raw_mark.strip().lower())
        candidate["mark"] = MARK_ALIASES.get(mark_key, mark_key)
    encoding = candidate.get("encoding")
    if not isinstance(encoding, dict):
        candidate["encoding"] = {}
    else:
        clean_encoding: dict[str, dict[str, Any]] = {}
        for channel, channel_value in encoding.items():
            if not isinstance(channel_value, dict):
                continue
            clean_channel = {
                str(key): value
                for key, value in channel_value.items()
                if value is not None and value != "" and value != [] and value != {}
            }
            if clean_channel:
                clean_encoding[str(channel)] = clean_channel
        candidate["encoding"] = clean_encoding
    if not isinstance(candidate.get("transform"), list) or not candidate.get("transform"):
        candidate.pop("transform", None)
    normalized = _normalize_nested(candidate)
    return normalized if isinstance(normalized, dict) else None


def canonical_key(spec: Any) -> str | None:
    normalized = normalize_spec(spec)
    return stable_json(normalized) if normalized is not None else None


def dedupe_specs(specs: Iterable[Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for spec in specs:
        normalized = normalize_spec(spec)
        if normalized is None:
            continue
        key = stable_json(normalized)
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def exact_match(pred: Any, gold: Any) -> bool:
    pred_key = canonical_key(pred)
    return pred_key is not None and pred_key == canonical_key(gold)


def _filter_tokens(spec: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    transforms = spec.get("transform", [])
    if not isinstance(transforms, list):
        return tokens
    for transform in transforms:
        if isinstance(transform, dict) and "filter" in transform:
            tokens.add(stable_json(_normalize_nested(transform["filter"])))
    return tokens


def spec_components(spec: Any) -> dict[str, set[str]]:
    normalized = normalize_spec(spec) or {}
    mark = normalized.get("mark")
    encoding = normalized.get("encoding", {})
    fields: set[str] = set()
    operations: set[str] = set()
    channels: set[str] = set()
    if isinstance(encoding, dict):
        for channel, value in encoding.items():
            if not isinstance(value, dict):
                continue
            channels.add(str(channel))
            field = value.get("field")
            if isinstance(field, str):
                fields.add(f"{channel}:{field.lower()}")
            elif field is None and value.get("aggregate") == "count":
                fields.add(f"{channel}:__count__")
            for operation in OPERATIONS:
                if operation in value:
                    operations.add(f"{channel}:{operation}:{stable_json(value[operation])}")
    return {
        "mark": {str(mark)} if mark is not None else set(),
        "channels": channels,
        "fields": fields,
        "operations": operations,
        "filters": _filter_tokens(normalized),
    }


def set_f1(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def graded_similarity(pred: Any, gold: Any) -> dict[str, float]:
    pred_components = spec_components(pred)
    gold_components = spec_components(gold)
    scores = {
        name: set_f1(pred_components[name], gold_components[name])
        for name in ("mark", "channels", "fields", "operations", "filters")
    }
    scores["macro"] = sum(scores.values()) / 5.0
    return scores


def best_graded_match(pred: Any, golds: list[dict[str, Any]]) -> dict[str, float]:
    if not golds:
        return {name: 0.0 for name in ("mark", "channels", "fields", "operations", "filters", "macro")}
    candidates = [graded_similarity(pred, gold) for gold in golds]
    return max(candidates, key=lambda item: item["macro"])


def ranked_exact_metrics(predictions: list[dict[str, Any]], golds: list[dict[str, Any]], ks: Iterable[int]) -> dict[str, float]:
    pred_keys = [canonical_key(spec) for spec in predictions]
    gold_keys = {canonical_key(spec) for spec in golds}
    output: dict[str, float] = {}
    first_rank = 0
    for rank, key in enumerate(pred_keys, start=1):
        if key is not None and key in gold_keys:
            first_rank = rank
            break
    output["MRR"] = 1.0 / first_rank if first_rank else 0.0
    for k in ks:
        prefix = pred_keys[:k]
        matched = {key for key in prefix if key is not None and key in gold_keys}
        output[f"Hit@{k}"] = float(bool(matched))
        output[f"Recall@{k}"] = len(matched) / len(gold_keys) if gold_keys else 0.0
        output[f"Precision@{k}"] = len(matched) / len(prefix) if prefix else 0.0
        p, r = output[f"Precision@{k}"], output[f"Recall@{k}"]
        output[f"F1@{k}"] = 2 * p * r / (p + r) if p + r else 0.0
    return output


def ndcg_at_k(predictions: list[dict[str, Any]], golds: list[dict[str, Any]], k: int) -> float:
    gains = [best_graded_match(spec, golds)["macro"] for spec in predictions[:k]]
    dcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sorted(gains + [1.0] * max(0, min(k, len(golds)) - len(gains)), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}


def extract_first_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    starts = [position for position in (cleaned.find("["), cleaned.find("{")) if position >= 0]
    for start in sorted(starts):
        opener = cleaned[start]
        closer = "]" if opener == "[" else "}"
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(cleaned)):
            char = cleaned[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : position + 1])
                    except json.JSONDecodeError:
                        break
    return None
