from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
AUDIT = HERE / "outputs" / "legacy_audit" / "legacy_audit_summary.json"
DETAIL = HERE / "outputs" / "legacy_audit" / "legacy_pool_stages.csv"
ABLATION = HERE / "out" / "a" / "step_ablation" / "summary.csv"
OUT = HERE / "out" / "fig"


def main() -> None:
    summary = json.loads(AUDIT.read_text(encoding="utf-8"))
    test = next(item for item in summary["split_audits"] if item["split"] == "test")
    rates = test["rates"]
    curves = summary["complete_pool_coverage_curves"]
    ks = curves["ks"]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    labels = ["Step 4", "Step 5", "Step 6", "Complete pool", "Heuristic top-5"]
    values = [
        rates["step4_contains_any_gold"], rates["step5_contains_any_gold"],
        rates["step6_contains_any_gold"], rates["complete_pool_any_gold"],
        rates["heuristic_top5_any_gold"],
    ]
    axes[0].bar(range(len(labels)), values, color=["#9ecae1", "#6baed6", "#3182bd", "#756bb1", "#31a354"])
    axes[0].set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Records containing ≥1 exact gold")
    axes[0].set_title("Privileged-stage coverage")
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.025, f"{100*value:.1f}%", ha="center", va="bottom", fontsize=8)

    for name, color, marker in (("heuristic", "#31a354", "o"), ("learned", "#d95f0e", "s")):
        metric = curves["metrics"][name]
        axes[1].plot(ks, [metric[f"Hit@{k}"] for k in ks], label=name.capitalize(), color=color, marker=marker, ms=3)
    axes[1].axhline(rates["complete_pool_any_gold"], color="#756bb1", linestyle="--", label="Complete-pool oracle")
    axes[1].set_xlim(1, 15)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Hit@K")
    axes[1].set_title("Fixed complete-pool ranking")
    axes[1].legend(frameon=False, loc="lower right")

    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / "legacy_coverage.png", dpi=300)
    figure.savefig(OUT / "legacy_coverage.pdf")
    plt.close(figure)

    with DETAIL.open("r", encoding="utf-8", newline="") as handle:
        cases = [row for row in csv.DictReader(handle) if row["record_id"].startswith("nvbench2:test:")]
    columns = [
        ("raw_count", "Raw 6+5+4"), ("expanded_count", "After expansion"),
        ("deduplicated_complete_count", "After deduplication"), ("cap15_count", "Legacy cap-15"),
    ]
    figure, axis = plt.subplots(figsize=(5.8, 3.0), constrained_layout=True)
    axis.boxplot(
        [[int(row[column]) for row in cases] for column, _ in columns],
        tick_labels=[label for _, label in columns], showfliers=False, patch_artist=True,
        boxprops={"facecolor": "#9ecae1", "edgecolor": "#2171b5"},
        medianprops={"color": "#cb181d"},
    )
    axis.set_ylabel("Candidates per record")
    axis.set_title("Candidate-pool stages (test, n=751)")
    axis.tick_params(axis="x", rotation=20)
    figure.savefig(OUT / "legacy_pool_sizes.png", dpi=300)
    figure.savefig(OUT / "legacy_pool_sizes.pdf")
    plt.close(figure)

    table_rows = [
        {"quantity": "step6_set_equals_gold", "value": rates["step6_set_equal_gold"]},
        {"quantity": "step6_contains_any_gold", "value": rates["step6_contains_any_gold"]},
        {"quantity": "complete_pool_oracle", "value": rates["complete_pool_any_gold"]},
        {"quantity": "legacy_heuristic_hit5", "value": summary["legacy_table_reproduction"]["heuristic"]["Hit@5"]},
        {"quantity": "legacy_learned_hit1", "value": summary["legacy_table_reproduction"]["learned"]["Hit@1"]},
        {"quantity": "legacy_learned_hit5", "value": summary["legacy_table_reproduction"]["learned"]["Hit@5"]},
    ]
    with (OUT / "legacy_key_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["quantity", "value"])
        writer.writeheader()
        writer.writerows(table_rows)

    with ABLATION.open("r", encoding="utf-8-sig", newline="") as handle:
        ablation_rows = list(csv.DictReader(handle))
    mode_order = ["full"] + [f"leave_step_{index}_out" for index in range(1, 7)]
    mode_labels = ["Full"] + [f"−S{index}" for index in range(1, 7)]
    rankers = [("heuristic", "#31a354"), ("learned", "#d95f0e"), ("oracle", "#756bb1")]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    width = 0.24
    for axis, metric in zip(axes, ("Hit@1", "Hit@5"), strict=True):
        for ranker_index, (ranker, color) in enumerate(rankers):
            lookup = {(row["step_mode"], row["ranker"]): float(row[metric]) for row in ablation_rows}
            positions = [index + (ranker_index - 1) * width for index in range(len(mode_order))]
            axis.bar(positions, [lookup[(mode, ranker)] for mode in mode_order], width=width, label=ranker.capitalize(), color=color)
        axis.set_xticks(range(len(mode_order)), mode_labels)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel(metric)
        axis.set_title(f"Answer-field ablation: {metric}")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, legend_labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    figure.savefig(OUT / "legacy_step_ablations.png", dpi=300)
    figure.savefig(OUT / "legacy_step_ablations.pdf")
    plt.close(figure)
    print(f"Wrote audit figures and table to {OUT}")


if __name__ == "__main__":
    main()
