#!/usr/bin/env python3
"""Source-level target-provenance audit of the public nvBench 2.0 SFT workflow."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import string
import subprocess
from pathlib import Path
from typing import Any


FILES = {
    "inference": "code/model_finetune/sft/infer.py",
    "templates": "code/model_finetune/sft/sft_template.py",
    "evaluation": "code/evaluation/evaluation.py",
}


def git_text(repository: Path, commit: str, relative: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "show", f"{commit}:{relative}"], text=True
    )


def format_keywords(tree: ast.AST) -> list[str]:
    fields: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "format":
            continue
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "PROMPT_TEMPLATE":
            fields.update(keyword.arg for keyword in node.keywords if keyword.arg)
    return sorted(fields)


def loaded_names_in_function(tree: ast.Module, function_name: str) -> list[str]:
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    return sorted({node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)})


def literal_assignments(tree: ast.Module) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            output[node.targets[0].id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    return output


def placeholders(value: str) -> list[str]:
    return sorted({name for _, name, _, _ in string.Formatter().parse(value) if name})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = {name: git_text(args.repository, args.commit, path) for name, path in FILES.items()}
    infer_tree = ast.parse(sources["inference"])
    template_tree = ast.parse(sources["templates"])
    assignments = literal_assignments(template_tree)
    format_fields = format_keywords(infer_tree)
    loaded = loaded_names_in_function(infer_tree, "dataset_preprocess")
    template_fields = {
        name: placeholders(value)
        for name, value in assignments.items()
        if name in {"SFT_PROMPT_TEMPLATE", "SFT_PROMPT_STEP_TEMPLATE"} and isinstance(value, str)
    }

    evaluator_separates_gold = (
        'ground_truth = json.loads(data.get("gold_answer", "[]"))' in sources["evaluation"]
        and '"model_predict": model_predict' in sources["evaluation"]
        and '"ground_truth": ground_truth' in sources["evaluation"]
    )
    inference_output_is_empty = 'output=""' in sources["inference"]
    gold_injected = "gold_answer" in format_fields or any(
        "gold_answer" in fields for fields in template_fields.values()
    )

    report = {
        "audit_type": "immutable source-level target-provenance audit",
        "repository": "https://github.com/HKUSTDial/nvBench-2.0",
        "commit": args.commit,
        "files": {
            name: {
                "path": FILES[name],
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            for name, text in sources.items()
        },
        "inference_prompt_format_fields": format_fields,
        "template_placeholders": template_fields,
        "gold_answer_loaded_by_preprocessor": "gold_answer_list" in loaded,
        "gold_answer_injected_into_inference_prompt": gold_injected,
        "inference_output_placeholder_is_empty": inference_output_is_empty,
        "evaluation_keeps_prediction_and_gold_in_separate_fields": evaluator_separates_gold,
        "per_instance_target_exposure_found": bool(gold_injected),
        "interpretation": (
            "The public SFT inference path serializes table columns, examples, unique-value counts, and the natural-language query. "
            "Although the dataset loader carries gold_answer through preprocessing, it is not passed to PROMPT_TEMPLATE.format; "
            "the output placeholder is empty. The evaluator reads gold answers only after predictions are produced."
        ),
        "boundary": "This audit establishes source-level input separation for the checked public SFT path; it does not reproduce trained-model performance or rule out training-data memorization.",
    }
    if gold_injected or not inference_output_is_empty or not evaluator_separates_gold:
        raise SystemExit("Public-workflow provenance assertions failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
