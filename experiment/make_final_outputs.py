from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "final"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: Any) -> str:
    return f"{100 * float(value):.2f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    legacy = json.loads((HERE / "outputs" / "legacy_audit" / "legacy_audit_summary.json").read_text(encoding="utf-8"))
    direct_ir = json.loads((HERE / "outputs" / "direct_ir" / "audit.json").read_text(encoding="utf-8"))
    ablation = read_csv(HERE / "out" / "a" / "step_ablation" / "summary.csv")
    cross = read_csv(HERE / "out" / "a" / "cross" / "weighted.csv")
    paired = read_csv(HERE / "out" / "a" / "cross" / "paired.csv")
    full = read_csv(HERE / "out" / "a" / "full" / "summary.csv")
    full_by_chart_family = read_csv(HERE / "out" / "a" / "full" / "by_chart_family.csv")
    full_by_task_operation = read_csv(HERE / "out" / "a" / "full" / "by_task_operation.csv")
    stability = read_csv(HERE / "out" / "a" / "stability" / "summary.csv")
    rerank = read_csv(HERE / "out" / "a" / "rerank" / "summary_design_weighted.csv")
    validation = json.loads((HERE / "out" / "validation.json").read_text(encoding="utf-8"))

    OUT.mkdir(parents=True, exist_ok=True)
    canonical = {
        "validation": validation,
        "legacy_table": legacy["legacy_table_reproduction"],
        "legacy_step6_test": next(item for item in legacy["split_audits"] if item["split"] == "test")["rates"],
        "step_ablations": ablation,
        "cross_family_design_weighted": cross,
        "paired_direct_staged_design_weighted": paired,
        "qwen_full_test": full,
        "qwen_full_by_chart_family": full_by_chart_family,
        "qwen_full_by_task_operation": full_by_task_operation,
        "stability": stability,
        "reranker_design_weighted": rerank,
        "direct_ir_conclusion": direct_ir["audit_conclusion"],
        "expert_status": "deferred; legacy synthetic ratings excluded",
    }
    (OUT / "canonical_results.json").write_text(json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8")

    cross_sorted = sorted(cross, key=lambda row: (row["condition"], row["model"]))
    cross_table = [
        [row["model"], row["condition"], row["n"], pct(row["any_valid_candidate"]), pct(row["raw_hit@1"]),
         pct(row["raw_hit@5"]), f"{float(row['top1_macro']):.3f}", f"{float(row['best5_macro']):.3f}",
         f"{float(row['elapsed_seconds']):.2f}"]
        for row in cross_sorted
    ]
    full_table = [
        [row["condition"], row["n"], pct(row["any_valid_candidate"]), pct(row["raw_hit@1"]),
         pct(row["raw_hit@5"]), f"{float(row['raw_mrr']):.3f}", f"{float(row['top1_macro']):.3f}"]
        for row in sorted(full, key=lambda row: row["condition"])
    ]
    rerank_table = [
        [row["model"], row["n"], f"{float(row['raw_union_count']):.2f}", f"{float(row['pool_size']):.2f}",
         pct(row["complete_pool_oracle"]), pct(row["rrf_hit@1"]), pct(row["rrf_hit@5"]),
         pct(row["llm_hit@1"]), pct(row["llm_hit@5"]), f"{float(row['llm_top1_graded']):.3f}"]
        for row in rerank
    ]
    step6 = canonical["legacy_step6_test"]
    legacy_rows = legacy["legacy_table_reproduction"]
    narrative = [
        "# Canonical machine-experiment results",
        "",
        f"Integrity status: **{validation['status']}**. Expert evidence is not included; the legacy synthetic ratings remain excluded.",
        "",
        "## Legacy privileged diagnostic",
        "",
        f"On 751 test records, `step_6.answer` equals the canonical gold set in {pct(step6['step6_set_equal_gold'])}% "
        f"and contains at least one gold in {pct(step6['step6_contains_any_gold'])}%. The privileged learned reranker reaches "
        f"Hit@1={pct(legacy_rows['learned']['Hit@1'])}% and Hit@5={pct(legacy_rows['learned']['Hit@5'])}%; these are not end-to-end results.",
        "",
        "## Cross-family forward generation (design-weighted 150-case sample)",
        "",
        md_table(["Model", "Condition", "n", "Any valid %", "Hit@1 %", "Hit@5 %", "Top-1 graded", "Best-5 graded", "Seconds/case"], cross_table),
        "",
        "## Qwen-14B full 751-case forward test",
        "",
        md_table(["Condition", "n", "Any valid %", "Hit@1 %", "Hit@5 %", "MRR", "Top-1 graded"], full_table),
        "",
        "## Leakage-free pooled reranking (design weighted)",
        "",
        md_table(["Reranker", "n", "Raw union", "Valid pool", "Pool oracle %", "RRF H@1 %", "RRF H@5 %", "LLM H@1 %", "LLM H@5 %", "LLM graded"], rerank_table),
        "",
        "## Direct-IR disposition",
        "",
        f"Strict structured generation is unsupported: {direct_ir['audit_conclusion']['reason']} The direct-IR branch is excluded from the revised scope.",
        "",
        "## Expert evidence",
        "",
        "Deferred by instruction. No synthetic expert rating is used in these results.",
    ]
    (OUT / "experiment_results.md").write_text("\n".join(narrative) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    direct = [row for row in cross_sorted if row["condition"] == "direct"]
    labels = [row["model"].split(":")[0] for row in direct]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    axes[0].bar(labels, [100 * float(row["raw_hit@5"]) for row in direct], color="#3182bd")
    axes[0].set_ylabel("Exact Hit@5 (%)")
    axes[0].set_title("Forward exact match")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, [float(row["top1_macro"]) for row in direct], color="#31a354")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Top-1 graded macro")
    axes[1].set_title("Forward near-valid fidelity")
    axes[1].tick_params(axis="x", rotation=25)
    figure.savefig(OUT / "cross_family_forward.png", dpi=300)
    figure.savefig(OUT / "cross_family_forward.pdf")
    plt.close(figure)

    if rerank:
        labels = [row["model"].split(":")[0] for row in rerank]
        x = list(range(len(labels)))
        width = 0.34
        figure, axis = plt.subplots(figsize=(5.2, 3.1), constrained_layout=True)
        axis.bar([value - width / 2 for value in x], [100 * float(row["rrf_hit@1"]) for row in rerank], width, label="RRF", color="#6baed6")
        axis.bar([value + width / 2 for value in x], [100 * float(row["llm_hit@1"]) for row in rerank], width, label="LLM", color="#e6550d")
        for value, row in zip(x, rerank, strict=True):
            axis.hlines(100 * float(row["complete_pool_oracle"]), value - 0.42, value + 0.42, color="#756bb1", linestyle="--")
        axis.set_xticks(x, labels)
        axis.set_ylabel("Exact Hit@1 (%)")
        axis.set_title("Leakage-free pool ordering")
        axis.legend(frameon=False)
        figure.savefig(OUT / "forward_reranking.png", dpi=300)
        figure.savefig(OUT / "forward_reranking.pdf")
        plt.close(figure)
    print(f"Wrote canonical results to {OUT}")


if __name__ == "__main__":
    main()
