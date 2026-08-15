from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    ablation = read_csv(HERE / "out" / "a" / "step_ablation" / "summary.csv")
    modes = ["full"] + [f"leave_step_{i}_out" for i in range(1, 7)]
    labels = ["Full"] + [f"−S{i}" for i in range(1, 7)]
    rankers = [("heuristic", "#31a354"), ("learned", "#d95f0e"), ("oracle", "#756bb1")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    width = 0.24
    for ax, metric in zip(axes, ("Hit@1", "Hit@5")):
        lookup = {(row["step_mode"], row["ranker"]): float(row[metric]) for row in ablation}
        for j, (ranker, color) in enumerate(rankers):
            xs = [i + (j - 1) * width for i in range(len(modes))]
            ax.bar(xs, [lookup[(m, ranker)] for m in modes], width, color=color, label=ranker.capitalize())
        ax.set_xticks(range(len(modes)), labels)
        ax.set_ylim(0, 1.05); ax.set_ylabel(metric); ax.set_title(f"Answer-field ablation: {metric}")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.savefig(ROOT / "figure3_privileged_step_ablations.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figure3_privileged_step_ablations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    cross = read_csv(HERE / "out" / "a" / "cross" / "weighted.csv")
    cross += [row for row in read_csv(HERE / "out" / "a" / "major_forward" / "weighted.csv") if row["model"] == "llama3.2:3b" and row["seed"] == "55"]
    cross += read_csv(HERE / "out" / "a" / "rich" / "summary_design_weighted.csv")
    models = ["gemma3:27b", "llama3.2:3b", "mistral-small:24b", "qwen3:14b"]
    model_labels = ["Gemma 27B", "Llama 3B", "Mistral 24B", "Qwen 14B"]
    lookup = {(row["model"], row["condition"]): row for row in cross}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    x = list(range(len(models))); width = 0.34
    for j, (condition, label, color) in enumerate((("direct", "Direct-basic", "#6baed6"), ("staged", "Staged-rich", "#e6550d"))):
        offset = -width / 2 if j == 0 else width / 2
        axes[0].bar([v + offset for v in x], [100 * float(lookup[(m, condition)]["raw_hit@1"]) for m in models], width, label=label, color=color)
        axes[1].bar([v + offset for v in x], [float(lookup[(m, condition)]["top1_macro"]) for m in models], width, label=label, color=color)
    rich = lookup[("qwen3:14b", "direct_rich")]
    axes[0].scatter([x[-1]], [100 * float(rich["raw_hit@1"])], marker="D", s=34, color="#756bb1", label="Qwen direct-rich", zorder=3)
    axes[1].scatter([x[-1]], [float(rich["top1_macro"])], marker="D", s=34, color="#756bb1", label="Qwen direct-rich", zorder=3)
    for ax in axes:
        ax.set_xticks(x, model_labels, rotation=15, ha="right")
    axes[0].set_ylabel("Exact Hit@1 (%)"); axes[0].set_title("Exact reproduction")
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Top-1 graded macro"); axes[1].set_title("Near-valid structural fidelity")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.savefig(ROOT / "figure4_forward_cross_family.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figure4_forward_cross_family.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pools = read_csv(HERE / "out" / "major_audit" / "three_pool_summary.csv")
    labels = ["Direct only", "Staged only", "Direct + staged"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2), constrained_layout=True)
    xs = range(3); width = 0.25
    ax.bar([x-width for x in xs], [100*float(r["oracle_any_gold"]) for r in pools], width, label="Pool oracle", color="#756bb1")
    ax.bar(xs, [100*float(r["rrf_hit1"]) for r in pools], width, label="RRF Hit@1", color="#31a354")
    ax.bar([x+width for x in xs], [100*float(r["rrf_hit5"]) for r in pools], width, label="RRF Hit@5", color="#6baed6")
    ax.set_xticks(list(xs), labels); ax.set_ylabel("Design-weighted exact rate (%)"); ax.set_title("Target-answer-free candidate-pool comparison")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.savefig(ROOT / "figure5_forward_reranking.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figure5_forward_reranking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
