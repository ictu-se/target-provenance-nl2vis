#!/usr/bin/env python3
"""Create the locked-holdout comparison figure from analyzed results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "gemma3:27b": "Gemma 3\n27B",
    "llama3.2:3b": "Llama 3.2\n3B",
    "mistral-small:24b": "Mistral-Small\n24B",
    "qwen3:14b": "Qwen 3\n14B",
}
CONDITIONS = ("direct", "direct_rich", "staged")
CONDITION_LABELS = ("Direct-basic", "Direct-rich", "Staged-rich")
COLORS = ("#4477AA", "#EE9944", "#228833")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["model"], row["condition"]): row for row in rows}
    models = [model for model in LABELS if all((model, condition) in lookup for condition in CONDITIONS)]
    if len(models) != 4:
        raise SystemExit(f"Expected four complete models, found {models}")

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), constrained_layout=True)
    x = np.arange(len(models))
    width = 0.23
    for axis, metric, title in (
        (axes[0], "raw_hit@1", "Full exact specification reproduction"),
        (axes[1], "core_hit@1", "Core-specification equivalence"),
    ):
        for offset, (condition, label, color) in enumerate(zip(CONDITIONS, CONDITION_LABELS, COLORS)):
            values = [100 * float(lookup[(model, condition)][metric]) for model in models]
            axis.bar(x + (offset - 1) * width, values, width, label=label, color=color)
        axis.set_title(title)
        axis.set_ylabel("Hit@1 (%)")
        axis.set_xticks(x, [LABELS[model] for model in models])
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=1, loc="upper left")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
