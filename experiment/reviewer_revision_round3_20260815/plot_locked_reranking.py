#!/usr/bin/env python3
"""Plot locked-holdout candidate-pool coverage and top-one ordering."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["model"], row["pool_name"]): row for row in rows}
    pools = ["direct", "staged", "union"]
    pool_labels = ["Direct pool", "Staged pool", "Union pool"]
    qwen = "qwen3:14b"
    mistral = "mistral-small:24b"
    series = [
        ("Pool oracle", [100 * float(lookup[(qwen, pool)]["complete_pool_oracle"]) for pool in pools], "#4C78A8"),
        ("RRF Hit@1", [100 * float(lookup[(qwen, pool)]["rrf_hit@1"]) for pool in pools], "#8FA7B8"),
        ("Mistral Hit@1", [100 * float(lookup[(mistral, pool)]["llm_hit@1"]) for pool in pools], "#D78345"),
        ("Qwen Hit@1", [100 * float(lookup[(qwen, pool)]["llm_hit@1"]) for pool in pools], "#8F6F9F"),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
    })
    fig, axis = plt.subplots(figsize=(6.8, 3.0))
    x = np.arange(len(pools))
    width = 0.19
    for index, (label, values, color) in enumerate(series):
        bars = axis.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}", ha="center", va="bottom", fontsize=7.7)
    axis.set_xticks(x, pool_labels)
    axis.set_ylabel("Design-weighted percentage")
    axis.set_ylim(0, max(value for _, values, _ in series for value in values) + 8)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
