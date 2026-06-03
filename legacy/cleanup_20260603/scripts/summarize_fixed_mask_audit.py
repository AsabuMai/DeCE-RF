#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = [
    "base_only",
    "direct_target",
    "adaptive_full_generic_support",
    "support_v3_fixed",
    "support_v3_controller_rmsgap",
]

TASK_ORDER = [
    "cat_crown",
    "bowl_apple_inside",
    "tshirt_star",
    "red_chair_blue",
    "pillow_vertical_fabric_strip",
    "backpack_remove_toy_charm",
]

METHOD_DISPLAY = {
    "base_only": "RF reconstruction / base reconstruction",
    "direct_target": "Direct target guidance",
    "adaptive_full_generic_support": "Generic support control",
    "support_v3_fixed": "Fixed DeCE displacement",
    "support_v3_controller_rmsgap": "DeCE-RF",
}

METRICS = [
    "fixed_eval_mask_area",
    "outside_mask_l1",
    "inside_mask_l1",
    "source_ssim",
    "dino_source_similarity",
    "edit_score",
]


def fnum(value: str) -> float:
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else math.nan


def stdev(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if len(clean) < 2:
        return 0.0
    mu = mean(clean)
    return math.sqrt(sum((value - mu) ** 2 for value in clean) / (len(clean) - 1))


def fmt(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("complete") == "True":
            groups[(row["task"], row["method"])].append(row)

    out_rows: list[dict[str, str]] = []
    tasks = sorted({key[0] for key in groups}, key=lambda task: (TASK_ORDER.index(task) if task in TASK_ORDER else 999, task))
    for task in tasks:
        for method in METHOD_ORDER:
            group = groups.get((task, method), [])
            if not group:
                continue
            summary = {
                "task": task,
                "method": method,
                "paper_name": METHOD_DISPLAY.get(method, method),
                "n": str(len(group)),
                "seeds": " ".join(sorted(row["seed"] for row in group)),
                "mask_source": group[0].get("mask_source", ""),
            }
            for metric in METRICS:
                values = [fnum(row.get(metric, "")) for row in group]
                summary[f"{metric}_mean"] = fmt(mean(values))
                summary[f"{metric}_std"] = fmt(stdev(values))
            out_rows.append(summary)
    return out_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "method",
        "paper_name",
        "n",
        "seeds",
        "mask_source",
        *[f"{metric}_{kind}" for metric in METRICS for kind in ("mean", "std")],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "Fixed per-task evaluation masks are reused across all methods and seeds.",
        "",
        "| Task | Method | n | Mask | Outside L1 | Inside L1 | Source SSIM | DINO/source | Edit score |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {task} | {paper_name} | {n} | {mask_source} | {outside} +- {outside_std} | "
            "{inside} +- {inside_std} | {ssim} +- {ssim_std} | {dino} +- {dino_std} | {edit} +- {edit_std} |".format(
                task=row["task"],
                paper_name=row["paper_name"],
                n=row["n"],
                mask_source=row["mask_source"],
                outside=row["outside_mask_l1_mean"],
                outside_std=row["outside_mask_l1_std"],
                inside=row["inside_mask_l1_mean"],
                inside_std=row["inside_mask_l1_std"],
                ssim=row["source_ssim_mean"],
                ssim_std=row["source_ssim_std"],
                dino=row["dino_source_similarity_mean"],
                dino_std=row["dino_source_similarity_std"],
                edit=row["edit_score_mean"],
                edit_std=row["edit_score_std"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_visual_template(path: Path, rows: list[dict[str, str]], grid_dir: Path) -> None:
    fieldnames = [
        "task",
        "method",
        "paper_name",
        "seed",
        "edit_success_1_5",
        "source_preservation_1_5",
        "locality_1_5",
        "artifact_severity_1_5",
        "overall_1_5",
        "failure_flag",
        "result_image",
        "review_grid",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task": row["task"],
                    "method": row["method"],
                    "paper_name": METHOD_DISPLAY.get(row["method"], row["method"]),
                    "seed": row["seed"],
                    "result_image": row["result_image"],
                    "review_grid": str(grid_dir / f"core4_main_seed{row['seed']}_grid.png"),
                    "notes": "Internal visual audit only; not a user study.",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize fixed-mask metrics into paper audit tables.")
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--md-output", type=Path, required=True)
    parser.add_argument("--visual-template-output", type=Path, required=True)
    parser.add_argument("--grid-dir", type=Path, required=True)
    args = parser.parse_args()

    raw_rows = read_rows(args.metrics_csv)
    summary_rows = summarize(raw_rows)
    write_csv(args.csv_output, summary_rows)
    write_markdown(args.md_output, summary_rows, "Strict Core-6 Fixed-Mask Quantitative Audit")
    paper_methods = {"base_only", "direct_target", "adaptive_full_generic_support", "support_v3_controller_rmsgap"}
    visual_rows = [
        row
        for row in raw_rows
        if row.get("complete") == "True" and row["method"] in paper_methods
    ]
    write_visual_template(args.visual_template_output, visual_rows, args.grid_dir)
    print(args.csv_output)
    print(args.md_output)
    print(args.visual_template_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
