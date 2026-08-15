#!/usr/bin/env python3
"""Plot empirical locked-holdout exact and validity outcomes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "gemma3:27b": "Gemma 3\n27B",
    "llama3.2:3b": "Llama 3.2\n3B",
    "mistral-small:24b": "Mistral-Small\n24B",
    "qwen3:14b": "Qwen 3\n14B",
}
PROMPTS = ["direct", "direct_rich", "staged"]
PROMPT_LABELS = ["Direct-basic", "Direct-rich", "Staged-rich"]
COLORS = ["#5B7FA3", "#9A7AA0", "#D78345"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(row["model"], row["condition"]): row for row in rows}
    models = list(MODEL_LABELS)
    metrics = [
        ("raw_hit@1", "Exact Hit@1"),
        ("top1_validity", "Top-one validity"),
        ("valid_fraction", "Valid-list fraction"),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 8,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), sharey=True)
    x = np.arange(len(models))
    width = 0.24
    for axis, (metric, title) in zip(axes, metrics):
        for offset, (prompt, label, color) in enumerate(zip(PROMPTS, PROMPT_LABELS, COLORS)):
            values = [100 * float(by_key[(model, prompt)][metric]) for model in models]
            axis.bar(x + (offset - 1) * width, values, width, label=label, color=color)
        axis.set_title(title)
        axis.set_xticks(x, [MODEL_LABELS[model] for model in models])
        axis.set_ylim(0, 100)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Design-weighted percentage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
