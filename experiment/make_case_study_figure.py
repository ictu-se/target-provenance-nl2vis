#!/usr/bin/env python3
"""Render four manuscript case studies from retained data and model outputs."""

from __future__ import annotations

import csv
import io
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import vl_convert as vlc


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DATASET = WORKSPACE / "data_benchmarks/datasets/nvBench-2.0/data/nvbench2.0/test.json"
CSV_DIR = WORKSPACE / (
    "data_benchmarks/datasets/nvBench-2.0/data/database/database_csv/database_csv"
)
RUN_DIR = HERE / "experiments/reviewer_revision_20260807/out/f"
RUNS = {
    "qwen_direct": RUN_DIR / "qwen3_14b_613573_d_55_t_full751.jsonl",
    "qwen_staged": RUN_DIR / "qwen3_14b_613573_s_55_t_full751.jsonl",
    "llama_direct": RUN_DIR / "llama3_2_3b_925b2c_d_55_t_cross150.jsonl",
    "llama_staged": RUN_DIR / "llama3_2_3b_925b2c_s_55_t_cross150.jsonl",
}


def load_jsonl(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8") as handle:
        return {row["index"]: row for row in map(json.loads, handle)}


def coerce(value: str):
    if value == "":
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", value):
        return float(value)
    return value


def load_values(csv_file: str) -> list[dict]:
    with (CSV_DIR / csv_file).open(encoding="utf-8-sig", newline="") as handle:
        return [{key: coerce(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def inferred_type(field: str | None, values: list[dict], channel: str, definition: dict) -> str:
    if definition.get("aggregate") or channel in {"theta", "size"}:
        return "quantitative"
    observed = [row.get(field) for row in values if field and row.get(field) is not None]
    if observed and all(isinstance(value, (int, float)) for value in observed):
        return "quantitative"
    if observed and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)) for value in observed):
        return "temporal"
    return "nominal"


def renderable_spec(spec: dict, values: list[dict]) -> dict:
    """Add renderer metadata without changing marks, fields, aggregates, or filters."""
    rendered = json.loads(json.dumps(spec))
    for channel, definition in rendered.get("encoding", {}).items():
        if not isinstance(definition, dict):
            continue
        if rendered.get("mark") == "rect" and channel in {"x", "y"}:
            definition["type"] = "nominal"
        elif definition.get("type") in {None, "datetime", "timeUnit"}:
            definition["type"] = inferred_type(definition.get("field"), values, channel, definition)
        if channel == "color":
            definition.setdefault("legend", {"title": None, "orient": "bottom", "columns": 2})

    # Keep coach colors stable while suppressing unobserved legend entries after filtering.
    color = rendered.get("encoding", {}).get("color", {})
    if color.get("field") == "coach_name":
        names = ["Jameson Tomas", "Joe Fabbri", "Robert Chen", "James Wong", "Smith Brown"]
        shown = names[:3] if rendered.get("transform") else names
        palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
        color["scale"] = {"domain": shown, "range": palette[: len(shown)]}

    rendered.update(
        {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": values},
            "width": 235,
            "height": 180,
            "view": {"stroke": None},
            "config": {
                "background": "white",
                "axis": {"labelFontSize": 16, "titleFontSize": 17, "labelLimit": 105},
                "legend": {"labelFontSize": 16, "titleFontSize": 17, "labelLimit": 110},
            },
        }
    )
    return rendered


def render(spec: dict, values: list[dict]) -> bytes:
    return vlc.vegalite_to_png(renderable_spec(spec, values), scale=2)


def make_figure(filename: str, case: dict) -> None:
    # Match the manuscript aspect ratio before inclusion.  The earlier 13.8-inch
    # canvas was reduced too aggressively and made chart labels difficult to read.
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.6), facecolor="white")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.75, bottom=0.10, wspace=0.08)
    fig.suptitle(
        f'Case {case["index"]}: “{case["record"]["nl_query"]}”',
        fontsize=14,
        fontweight="semibold",
    )

    values = load_values(case["record"]["csv_file"])
    for column_index, panel in enumerate(case["panels"]):
        axis = axes[column_index]
        png = render(panel["spec"], values)
        axis.imshow(plt.imread(io.BytesIO(png)))
        axis.set_title(
            textwrap.fill(panel["title"], width=27),
            fontsize=13.5,
            fontweight="bold",
            pad=6,
        )
        axis.text(
            0.5,
            -0.04,
            textwrap.fill(panel["note"], width=29),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=12.5,
        )
        axis.axis("off")

    fig.savefig(HERE / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{filename}.png", dpi=300, bbox_inches="tight")


def main() -> None:
    records = json.loads(DATASET.read_text(encoding="utf-8"))
    runs = {name: load_jsonl(path) for name, path in RUNS.items()}

    def gold(index: int) -> list[dict]:
        return json.loads(records[index]["gold_answer"])

    cases = {
        14: {
            "index": 14,
            "record": records[14],
            "panels": [
                {
                    "title": "(a) Qwen direct top 1",
                    "spec": runs["qwen_direct"][14]["candidates"][0],
                    "note": "Filter omitted: all five coaches remain",
                },
                {
                    "title": "(b) Qwen staged top 1 = Gold 1",
                    "spec": runs["qwen_staged"][14]["candidates"][0],
                    "note": "Requested filter retained; equal counts",
                },
                {
                    "title": "(c) Benchmark Gold 2",
                    "spec": gold(14)[1],
                    "note": "Same filter; rank controls slice angle",
                },
            ],
        },
        77: {
            "index": 77,
            "record": records[77],
            "panels": [
                {
                    "title": "(a) Qwen direct top 1",
                    "spec": runs["qwen_direct"][77]["candidates"][0],
                    "note": "Aggregates collapse the requested mapping",
                },
                {
                    "title": "(b) Qwen staged top 1 = Gold 4",
                    "spec": runs["qwen_staged"][77]["candidates"][0],
                    "note": "Games, wins, league, and count are retained",
                },
                {
                    "title": "(c) Benchmark Gold 1",
                    "spec": gold(77)[0],
                    "note": "Alternative mapping uses rank and games",
                },
            ],
        },
        142: {
            "index": 142,
            "record": records[142],
            "panels": [
                {
                    "title": "(a) Qwen direct top 1 (non-exact)",
                    "spec": runs["qwen_direct"][142]["candidates"][0],
                    "note": "Same visible design; added field-type declarations",
                },
                {
                    "title": "(b) Qwen staged top 1 = Gold 1",
                    "spec": runs["qwen_staged"][142]["candidates"][0],
                    "note": "Workshop-group ID interpretation",
                },
                {
                    "title": "(c) Qwen staged top 2 = Gold 2",
                    "spec": runs["qwen_staged"][142]["candidates"][1],
                    "note": "Address-ID interpretation",
                },
            ],
        },
        358: {
            "index": 358,
            "record": records[358],
            "panels": [
                {
                    "title": "(a) Qwen direct top 1 = Gold 4",
                    "spec": runs["qwen_direct"][358]["candidates"][0],
                    "note": "Daily fee is summed by product type",
                },
                {
                    "title": "(b) Qwen staged top 1\n(normal-form noncompliant)",
                    "spec": runs["qwen_staged"][358]["candidates"][0],
                    "note": "Count-with-field yields equal cost slices",
                },
                {
                    "title": "(c) Benchmark Gold 1",
                    "spec": gold(358)[0],
                    "note": "Daily fee mapped to individual products",
                },
            ],
        },
    }

    for figure_number, index in enumerate((14, 77, 142, 358), start=6):
        make_figure(f"figure{figure_number}_rendered_case_{index}", cases[index])

    manifest = {
        "case_selection": (
            "Post-hoc explanatory cases covering filter retention, specification mapping, "
            "multiple acceptable interpretations, exact-match brittleness, and a staged counterexample."
        ),
        "rendering_note": (
            "Renderer-required type declarations and presentation properties were added; "
            "marks, fields, aggregates, and filters were not changed."
        ),
        "cases": [],
    }
    for index in [14, 77, 142, 358]:
        entry = {
            "record_id": f"nvbench2:test:{index}",
            "query": records[index]["nl_query"],
            "csv_file": records[index]["csv_file"],
            "panels": [
                {"title": panel["title"], "note": panel["note"], "spec": panel["spec"]}
                for panel in cases[index]["panels"]
            ],
        }
        manifest["cases"].append(entry)
    (HERE / "rendered_case_studies_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
