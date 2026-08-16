#!/usr/bin/env python3
"""Plot the empirical Hit@1 distribution across ten presentation orders."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "mistral-small:24b": "Mistral-Small 24B",
    "qwen3:14b": "Qwen 3 14B",
}
COLORS = {
    "mistral-small:24b": "#4C78A8",
    "qwen3:14b": "#F58518",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    models = ["mistral-small:24b", "qwen3:14b"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for axis, model in zip(axes, models):
        selected = sorted((row for row in rows if row["model"] == model), key=lambda row: int(row["permutation_seed"]))
        seeds = [int(row["permutation_seed"]) for row in selected]
        hit1 = [100 * float(row["hit1"]) for row in selected]
        rrf = 100 * float(selected[0]["rrf_hit1"])
        x = range(len(seeds))
        axis.plot(x, hit1, marker="o", linewidth=1.5, markersize=5, color=COLORS[model], label="Spec-only LLM")
        axis.axhline(rrf, color="#666666", linestyle="--", linewidth=1.4, label="RRF ensemble")
        axis.set_xticks(list(x), [str(seed) for seed in seeds], rotation=45)
        axis.set_title(LABELS[model])
        axis.set_xlabel("Presentation-permutation seed")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Design-weighted Hit@1 (%)")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
