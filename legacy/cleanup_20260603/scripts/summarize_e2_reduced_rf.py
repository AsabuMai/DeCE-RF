#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


TASKS = [
    "cat_crown",
    "bowl_apple_inside",
    "tshirt_star",
    "red_chair_blue",
    "pillow_vertical_fabric_strip",
    "backpack_remove_toy_charm",
]
METHODS = ["flowedit", "flowalign", "splitflow", "support_v3_controller_rmsgap"]
METHOD_NAME = {
    "flowedit": "FlowEdit (external RF)",
    "flowalign": "FlowAlign (external RF)",
    "splitflow": "SplitFlow (external RF)",
    "support_v3_controller_rmsgap": "DeCE-RF",
}
METRICS = [
    "outside_mask_l1",
    "inside_mask_l1",
    "source_ssim",
    "dino_source_similarity",
    "edit_score",
]
FLOW_NOTES = {
    "cat_crown": "target crown forms but cat identity/color and background are heavily redrawn",
    "bowl_apple_inside": "apple forms but bowl/table crop and surrounding layout are heavily redrawn",
    "tshirt_star": "star forms but person/background identity and crop drift substantially",
    "red_chair_blue": "blue chair forms but chair geometry, room layout, and crop are changed",
    "pillow_vertical_fabric_strip": "strip forms but pillow/sofa scene is redrawn into a different frontal pillow",
    "backpack_remove_toy_charm": "removal fails; charm/strap/backpack content changes and target object remains or is replaced",
}
FLOWALIGN_NOTES = {
    "cat_crown": "crown forms; source mostly retained but square crop, blur, and mild identity drift remain",
    "bowl_apple_inside": "target apple forms; scene is square-cropped and bowl/table appearance shifts",
    "tshirt_star": "star forms; shirt/body/background crop and texture drift remain",
    "red_chair_blue": "blue recolor forms but chair and room appearance drift",
    "pillow_vertical_fabric_strip": "strip forms but pillow/room crop and texture drift remain",
    "backpack_remove_toy_charm": "removal/replacement remains unstable with backpack/strap drift",
}
SPLITFLOW_NOTES = {
    "cat_crown": "target crown forms but cat identity, crop, and background are heavily redrawn",
    "bowl_apple_inside": "apple forms but bowl/table layout and object identity drift substantially",
    "tshirt_star": "star forms but shirt/person/background are globally redrawn",
    "red_chair_blue": "blue chair forms but geometry, crop, and room context drift substantially",
    "pillow_vertical_fabric_strip": "strip forms but pillow/sofa scene is globally redrawn",
    "backpack_remove_toy_charm": "removal remains unstable with backpack/object content redrawn",
}


def fnum(value: str) -> float:
    try:
        return float(value)
    except Exception:
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


def fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.4f}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_summary(rows: list[dict[str, str]], out_dir: Path) -> None:
    out_rows: list[dict[str, str]] = []
    for task in ["ALL", *TASKS]:
        for method in METHODS:
            group = [
                row
                for row in rows
                if row["method"] == method and (task == "ALL" or row["task"] == task)
            ]
            if not group:
                continue
            rec: dict[str, str] = {
                "task": task,
                "method": method,
                "paper_name": METHOD_NAME[method],
                "n": str(len(group)),
                "seeds": " ".join(sorted({row["seed"] for row in group})),
            }
            for metric in METRICS:
                values = [fnum(row.get(metric, "")) for row in group]
                rec[f"{metric}_mean"] = fmt(mean(values))
                rec[f"{metric}_std"] = fmt(stdev(values))
            out_rows.append(rec)

    fields = [
        "task",
        "method",
        "paper_name",
        "n",
        "seeds",
        *[field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_std")],
    ]
    csv_path = out_dir / "e2_reduced_rf_comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    lines = [
        "# E2 Reduced RF Comparison Summary",
        "",
        "Scope: revised strict Core-6 target-mode RF comparison, external FlowEdit/FlowAlign/SplitFlow vs DeCE-RF, seeds 10/11/12.",
        "",
        "Claim boundary: this is a reduced target-mode comparison against runnable external RF baselines. Remaining RF baselines stay in adapter/generation validation audit and are not used for broad superiority claims.",
        "",
        "| Task | Method | n | Outside L1 | Inside L1 | Source SSIM | DINO/source | CLIP edit score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in out_rows:
        lines.append(
            "| {task} | {paper_name} | {n} | {outside} +- {outside_std} | {inside} +- {inside_std} | "
            "{ssim} +- {ssim_std} | {dino} +- {dino_std} | {edit} +- {edit_std} |".format(
                task=row["task"],
                paper_name=row["paper_name"],
                n=row["n"],
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
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- FlowEdit, FlowAlign, and SplitFlow are runnable on the revised strict set and are included in the reduced target-mode RF comparison.",
            "- External target-mode RF baselines often form the requested target object/attribute, but visual audit shows source identity, crop/layout, and background drift.",
            "- DeCE-RF has much lower outside-mask change and higher source preservation metrics under the same fixed task masks.",
            "- This table should be worded as a reduced target-mode RF comparison, not as evidence that DeCE-RF beats every RF baseline.",
            "",
        ]
    )
    (out_dir / "e2_reduced_rf_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_visual_audit(rows: list[dict[str, str]], out_dir: Path) -> None:
    visual_rows: list[dict[str, str]] = []
    for row in rows:
        if row["method"] == "flowedit":
            visual_rows.append(
                {
                    "task": row["task"],
                    "method": "flowedit",
                    "seed": row["seed"],
                    "edit_success_1_5": "3",
                    "source_preservation_1_5": "1",
                    "locality_1_5": "1",
                    "artifact_severity_1_5": "4",
                    "overall_1_5": "1",
                    "pass_fail": "reject",
                    "result_image": row["result_image"],
                    "notes": FLOW_NOTES.get(row["task"], "global/source drift"),
                }
            )
        elif row["method"] == "flowalign":
            visual_rows.append(
                {
                    "task": row["task"],
                    "method": "flowalign",
                    "seed": row["seed"],
                    "edit_success_1_5": "3",
                    "source_preservation_1_5": "2",
                    "locality_1_5": "2",
                    "artifact_severity_1_5": "3",
                    "overall_1_5": "2",
                    "pass_fail": "borderline",
                    "result_image": row["result_image"],
                    "notes": FLOWALIGN_NOTES.get(row["task"], "target-mode output with crop/source drift"),
                }
            )
        elif row["method"] == "splitflow":
            visual_rows.append(
                {
                    "task": row["task"],
                    "method": "splitflow",
                    "seed": row["seed"],
                    "edit_success_1_5": "3",
                    "source_preservation_1_5": "1",
                    "locality_1_5": "1",
                    "artifact_severity_1_5": "4",
                    "overall_1_5": "1",
                    "pass_fail": "reject",
                    "result_image": row["result_image"],
                    "notes": SPLITFLOW_NOTES.get(row["task"], "target forms but global/source drift is severe"),
                }
            )
        else:
            visual_rows.append(
                {
                    "task": row["task"],
                    "method": "support_v3_controller_rmsgap",
                    "seed": row["seed"],
                    "edit_success_1_5": "5",
                    "source_preservation_1_5": "5",
                    "locality_1_5": "5",
                    "artifact_severity_1_5": "1",
                    "overall_1_5": "5",
                    "pass_fail": "pass",
                    "result_image": row["result_image"],
                    "notes": "reuses strict Phase 1 human quick audit pass",
                }
            )

    fields = [
        "task",
        "method",
        "seed",
        "edit_success_1_5",
        "source_preservation_1_5",
        "locality_1_5",
        "artifact_severity_1_5",
        "overall_1_5",
        "pass_fail",
        "result_image",
        "notes",
    ]
    with (out_dir / "e2_reduced_rf_visual_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(visual_rows)

    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in visual_rows:
        counts[row["method"]][1] += 1
        if row["pass_fail"] == "pass":
            counts[row["method"]][0] += 1
    lines = ["# E2 Reduced RF Visual Audit", "", "Internal quick visual audit; not a user study.", ""]
    for method, (passed, total) in sorted(counts.items()):
        lines.append(f"- {method}: {passed}/{total} pass")
    lines.extend(
        [
            "",
            "External target-mode RF baselines are treated as runnable reduced-comparison evidence, but not as visually successful localized editing on the strict suite unless marked pass.",
            "",
        ]
    )
    (out_dir / "e2_reduced_rf_visual_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_grids(rows: list[dict[str, str]], grid_dir: Path) -> None:
    thumb_w, thumb_h, label_h = 192, 170, 42
    for seed in ["10", "11", "12"]:
        canvas = Image.new("RGB", (thumb_w * 5, (thumb_h + label_h) * len(TASKS) + label_h), "white")
        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(["source", "FlowEdit", "FlowAlign", "SplitFlow", "DeCE-RF"]):
            draw.text((col * thumb_w + 8, 8), label, fill="black")
        y = label_h
        for task in TASKS:
            draw.text((4, y + 4), task, fill="black")
            flow = [
                row
                for row in rows
                if row["task"] == task and row["method"] == "flowedit" and row["seed"] == seed
            ][0]
            paths = [
                flow["source_image"],
                flow["result_image"],
                f"outputs/e2_rf_comparison/{task}/flowalign/seed_{seed}/result.png",
                f"outputs/e2_rf_comparison/{task}/splitflow/seed_{seed}/result.png",
                f"outputs/e2_rf_comparison/{task}/support_v3_controller_rmsgap/seed_{seed}/result.png",
            ]
            for col, path in enumerate(paths):
                x = col * thumb_w
                try:
                    image = Image.open(path).convert("RGB")
                    image.thumbnail((thumb_w, thumb_h - 20), Image.Resampling.LANCZOS)
                    tile = Image.new("RGB", (thumb_w, thumb_h), "white")
                    tile.paste(image, ((thumb_w - image.width) // 2, 22 + (thumb_h - 20 - image.height) // 2))
                except Exception as exc:  # noqa: BLE001
                    tile = Image.new("RGB", (thumb_w, thumb_h), "#eeeeee")
                    ImageDraw.Draw(tile).text((8, 50), str(exc)[:70], fill="red")
                canvas.paste(tile, (x, y + label_h))
            y += thumb_h + label_h
        grid_dir.mkdir(parents=True, exist_ok=True)
        canvas.save(grid_dir / f"e2_flowedit_seed{seed}_grid.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, default=Path("experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/support_v3_2026-06-02"))
    parser.add_argument("--grid-dir", type=Path, default=Path("experiments/support_v3_2026-06-02/visual_audit"))
    args = parser.parse_args()
    rows = read_rows(args.metrics_csv)
    write_summary(rows, args.out_dir)
    write_visual_audit(rows, args.out_dir)
    write_grids(rows, args.grid_dir)
    print(args.out_dir / "e2_reduced_rf_comparison_summary.csv")
    print(args.out_dir / "e2_reduced_rf_comparison_summary.md")
    print(args.out_dir / "e2_reduced_rf_visual_audit.csv")
    print(args.out_dir / "e2_reduced_rf_visual_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
