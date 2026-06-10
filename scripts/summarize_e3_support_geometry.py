from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


TASKS = ["cat_crown", "tshirt_star", "backpack_remove_toy_charm"]
SEEDS = ["10", "11", "12"]
EXP = Path("experiments/support_v3_2026-06-02")
OUT = EXP / "e3_support_geometry"
CANVAS = 512


VARIANTS = [
    {
        "variant": "attention_only",
        "display": "Attention only",
        "method": "support_v3_controller_rmsgap",
        "path": "operation_v3_candidate_attention_only.png",
        "threshold_policy": "top_5_percent_nonzero",
        "role": "semantic localization only",
    },
    {
        "variant": "clean_disagreement",
        "display": "Clean disagreement",
        "method": "support_v3_controller_rmsgap",
        "path": "operation_v3_candidate_clean_disagreement_only.png",
        "threshold_policy": "top_5_percent_nonzero",
        "role": "source-target clean-estimate gap",
    },
    {
        "variant": "velocity_disagreement",
        "display": "Velocity disagreement",
        "method": "support_v3_controller_rmsgap",
        "path": "generic_velocity_disagreement_map.png",
        "threshold_policy": "top_5_percent_nonzero",
        "role": "RF response without relation prior",
    },
    {
        "variant": "grounding_sam",
        "display": "Grounding/SAM",
        "method": "support_v3_controller_rmsgap",
        "path": "operation_v3_grounding_mask.png",
        "threshold_policy": "binary_midpoint",
        "role": "external segmentation only",
    },
    {
        "variant": "generic_support",
        "display": "Generic support",
        "method": "adaptive_full_generic_support",
        "path": "core_final.png",
        "threshold_policy": "binary_midpoint",
        "role": "weak automatic support",
        "downstream_method": "adaptive_full_generic_support",
    },
    {
        "variant": "operation_conditioned_support",
        "display": "Operation support",
        "method": "support_v3_controller_rmsgap",
        "path": "operation_v3_edit_mask.png",
        "threshold_policy": "binary_midpoint",
        "role": "operation-conditioned geometry estimator",
        "downstream_method": "support_v3_controller_rmsgap",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_gray(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("L")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def binarize(image: Image.Image, policy: str) -> Image.Image:
    values = list(image.getdata())
    if policy == "binary_midpoint":
        threshold = 127
    elif policy == "top_5_percent_nonzero":
        nonzero = sorted(v for v in values if v > 0)
        if not nonzero:
            threshold = 255
        else:
            idx = max(0, int(round(len(nonzero) * 0.95)) - 1)
            threshold = max(1, nonzero[idx])
    else:
        raise ValueError(f"unknown threshold policy {policy}")
    return image.point(lambda px: 255 if px >= threshold else 0)


def mask_metrics(pred: Image.Image, target: Image.Image) -> dict[str, float]:
    pred_values = [v > 0 for v in pred.getdata()]
    target_values = [v > 0 for v in target.getdata()]
    intersection = sum(p and t for p, t in zip(pred_values, target_values))
    union = sum(p or t for p, t in zip(pred_values, target_values))
    pred_area = sum(pred_values)
    target_area = sum(target_values)
    total = len(pred_values)
    return {
        "support_iou": intersection / union if union else 0.0,
        "support_precision": intersection / pred_area if pred_area else 0.0,
        "support_recall": intersection / target_area if target_area else 0.0,
        "support_area_ratio": pred_area / total if total else 0.0,
        "eval_area_ratio": target_area / total if total else 0.0,
    }


def index_downstream_metrics() -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_csv(EXP / "strict_fixed_mask_metrics.csv")
    by_key = {}
    for row in rows:
        key = (row.get("task", ""), row.get("method", ""), str(row.get("seed", "")))
        by_key[key] = row
    return by_key


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def make_rows() -> list[dict[str, str]]:
    downstream = index_downstream_metrics()
    rows = []
    for task in TASKS:
        eval_mask_path = EXP / "normalized_512" / "eval_masks" / f"{task}_eval_mask.png"
        eval_mask = binarize(load_gray(eval_mask_path), "binary_midpoint")
        for seed in SEEDS:
            for spec in VARIANTS:
                mask_path = (
                    Path("outputs")
                    / "pretty_matrix"
                    / task
                    / spec["method"]
                    / f"seed_{seed}"
                    / "masks"
                    / spec["path"]
                )
                row = {
                    "task": task,
                    "seed": seed,
                    "variant": spec["variant"],
                    "display_name": spec["display"],
                    "role": spec["role"],
                    "mask_path": str(mask_path),
                    "threshold_policy": spec["threshold_policy"],
                    "complete": "0",
                    "missing": "0",
                }
                if not mask_path.exists():
                    row.update(
                        {
                            "complete": "0",
                            "missing": "1",
                            "support_iou": "",
                            "support_precision": "",
                            "support_recall": "",
                            "support_area_ratio": "",
                            "eval_area_ratio": "",
                        }
                    )
                    rows.append(row)
                    continue
                support = binarize(load_gray(mask_path, eval_mask.size), spec["threshold_policy"])
                metrics = mask_metrics(support, eval_mask)
                row["complete"] = "1"
                for key, value in metrics.items():
                    row[key] = fmt(value)
                downstream_method = spec.get("downstream_method", "")
                metric_row = downstream.get((task, downstream_method, seed), {}) if downstream_method else {}
                row["downstream_method"] = downstream_method
                row["outside_mask_l1"] = metric_row.get("outside_mask_l1", "")
                row["inside_mask_l1"] = metric_row.get("inside_mask_l1", "")
                row["source_ssim_luma"] = metric_row.get("source_ssim_luma", "")
                row["edit_score"] = metric_row.get("edit_score", "")
                rows.append(row)
    return rows


def avg(rows: list[dict[str, str]], key: str) -> float | None:
    values = []
    for row in rows:
        try:
            if row.get(key, "") != "":
                values.append(float(row[key]))
        except ValueError:
            pass
    return sum(values) / len(values) if values else None


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def avg_fmt(rows: list[dict[str, str]], key: str) -> str:
    value = avg(rows, key)
    return fmt(value) if value is not None else ""


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["complete"] == "1":
            by_variant[row["variant"]].append(row)
    summary = []
    for spec in VARIANTS:
        group = by_variant[spec["variant"]]
        row = {
            "variant": spec["variant"],
            "display_name": spec["display"],
            "n": str(len(group)),
            "role": spec["role"],
            "threshold_policy": spec["threshold_policy"],
        }
        for key in [
            "support_iou",
            "support_precision",
            "support_recall",
            "support_area_ratio",
            "outside_mask_l1",
            "inside_mask_l1",
            "source_ssim_luma",
            "edit_score",
        ]:
            row[key + "_mean"] = avg_fmt(group, key)
        summary.append(row)
    return summary


def summarize_by_task(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["complete"] == "1":
            by_key[(row["task"], row["variant"])].append(row)
    summary = []
    for task in TASKS:
        for spec in VARIANTS:
            group = by_key[(task, spec["variant"])]
            row = {
                "task": task,
                "variant": spec["variant"],
                "display_name": spec["display"],
                "n": str(len(group)),
                "role": spec["role"],
            }
            for key in [
                "support_iou",
                "support_precision",
                "support_recall",
                "support_area_ratio",
                "outside_mask_l1",
                "inside_mask_l1",
                "source_ssim_luma",
                "edit_score",
            ]:
                row[key + "_mean"] = avg_fmt(group, key)
            summary.append(row)
    return summary


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def ranks(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        j = idx
        while j + 1 < len(order) and order[j + 1][1] == order[idx][1]:
            j += 1
        rank = (idx + j + 2) / 2.0
        for k in range(idx, j + 1):
            result[order[k][0]] = rank
        idx = j + 1
    return result


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(ranks(xs), ranks(ys))


def corr_fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def correlation_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    runnable = [
        row
        for row in rows
        if row.get("downstream_method")
        and row.get("outside_mask_l1", "") != ""
        and row.get("edit_score", "") != ""
    ]
    support_keys = [
        "support_iou",
        "support_precision",
        "support_recall",
        "support_area_ratio",
    ]
    downstream_keys = [
        "outside_mask_l1",
        "inside_mask_l1",
        "source_ssim_luma",
        "edit_score",
    ]
    out = []
    for support_key in support_keys:
        for downstream_key in downstream_keys:
            paired = []
            for row in runnable:
                try:
                    paired.append((float(row[support_key]), float(row[downstream_key])))
                except ValueError:
                    pass
            xs = [item[0] for item in paired]
            ys = [item[1] for item in paired]
            out.append(
                {
                    "support_metric": support_key,
                    "downstream_metric": downstream_key,
                    "n": str(len(paired)),
                    "pearson_r": corr_fmt(pearson(xs, ys)),
                    "spearman_r": corr_fmt(spearman(xs, ys)),
                    "scope": "runnable rows only: generic_support and operation_conditioned_support",
                }
            )
    return out


def colorize_mask(mask: Image.Image) -> Image.Image:
    gray = ImageOps.autocontrast(mask.convert("L"))
    rgb = Image.new("RGB", gray.size, (248, 248, 248))
    overlay = Image.new("RGB", gray.size, (220, 32, 32))
    return Image.blend(rgb, overlay, 0.65).convert("RGB").putalpha(gray) if False else ImageOps.colorize(gray, black="#f6f6f6", white="#d21f3c")


def fit_rgb(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = min(CANVAS / image.width, CANVAS / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    fitted = image.resize(size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 245))
    canvas.paste(fitted, ((CANVAS - size[0]) // 2, (CANVAS - size[1]) // 2))
    return canvas


def fit_mask(path: Path, policy: str | None = None) -> Image.Image:
    mask = load_gray(path)
    if policy:
        mask = binarize(mask, policy)
    image = colorize_mask(mask)
    return fit_pil(image)


def fit_pil(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    scale = min(CANVAS / image.width, CANVAS / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    fitted = image.resize(size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (CANVAS, CANVAS), (245, 245, 245))
    canvas.paste(fitted, ((CANVAS - size[0]) // 2, (CANVAS - size[1]) // 2))
    return canvas


def label_cell(image: Image.Image, label: str) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, CANVAS, 34), fill=(0, 0, 0))
    draw.text((8, 8), label, fill=(255, 255, 255), font=font)
    return canvas


def make_figure_panel(task: str = "tshirt_star", seed: str = "10") -> Path:
    run = Path("outputs") / "pretty_matrix" / task / "support_v3_controller_rmsgap" / f"seed_{seed}"
    generic = Path("outputs") / "pretty_matrix" / task / "adaptive_full_generic_support" / f"seed_{seed}"
    source = EXP / "normalized_512" / "sources" / f"{task}.png"
    eval_mask = EXP / "normalized_512" / "eval_masks" / f"{task}_eval_mask.png"
    cells = [
        ("source", fit_rgb(source)),
        ("eval mask", fit_mask(eval_mask, "binary_midpoint")),
        ("attention", fit_mask(run / "masks" / "operation_v3_candidate_attention_only.png")),
        ("clean gap", fit_mask(run / "masks" / "operation_v3_candidate_clean_disagreement_only.png")),
        ("velocity gap", fit_mask(run / "masks" / "generic_velocity_disagreement_map.png")),
        ("Grounding/SAM", fit_mask(run / "masks" / "operation_v3_grounding_mask.png", "binary_midpoint")),
        ("generic support", fit_mask(generic / "masks" / "core_final.png", "binary_midpoint")),
        ("operation M_edit", fit_mask(run / "masks" / "operation_v3_edit_mask.png", "binary_midpoint")),
        ("M_preserve", fit_mask(run / "masks" / "operation_v3_preserve_mask.png", "binary_midpoint")),
        ("DeCE-RF result", fit_rgb(run / "result.png")),
    ]
    cols = 5
    rows = 2
    grid = Image.new("RGB", (cols * CANVAS, rows * CANVAS), (245, 245, 245))
    for idx, (label, image) in enumerate(cells):
        x = (idx % cols) * CANVAS
        y = (idx // cols) * CANVAS
        grid.paste(label_cell(image, label), (x, y))
    path = OUT / f"e3_support_geometry_{task}_seed{seed}_figure4_panel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    return path


def make_seed10_sheet() -> Path:
    columns = [
        ("source", None),
        ("eval mask", ("eval", "binary_midpoint")),
        ("attention", ("operation_v3_candidate_attention_only.png", None)),
        ("clean gap", ("operation_v3_candidate_clean_disagreement_only.png", None)),
        ("velocity gap", ("generic_velocity_disagreement_map.png", None)),
        ("Grounding/SAM", ("operation_v3_grounding_mask.png", "binary_midpoint")),
        ("generic", ("generic_core_final", "binary_midpoint")),
        ("operation", ("operation_v3_edit_mask.png", "binary_midpoint")),
        ("result", None),
    ]
    grid = Image.new("RGB", (len(columns) * CANVAS, len(TASKS) * CANVAS), (245, 245, 245))
    for row_idx, task in enumerate(TASKS):
        run = Path("outputs") / "pretty_matrix" / task / "support_v3_controller_rmsgap" / "seed_10"
        generic = Path("outputs") / "pretty_matrix" / task / "adaptive_full_generic_support" / "seed_10"
        for col_idx, (label, spec) in enumerate(columns):
            if label == "source":
                image = fit_rgb(EXP / "normalized_512" / "sources" / f"{task}.png")
            elif label == "result":
                image = fit_rgb(run / "result.png")
            elif spec and spec[0] == "eval":
                image = fit_mask(EXP / "normalized_512" / "eval_masks" / f"{task}_eval_mask.png", spec[1])
            elif spec and spec[0] == "generic_core_final":
                image = fit_mask(generic / "masks" / "core_final.png", spec[1])
            else:
                image = fit_mask(run / "masks" / spec[0], spec[1] if spec else None)
            text = f"{task}: {label}" if col_idx == 0 else label
            grid.paste(label_cell(image, text), (col_idx * CANVAS, row_idx * CANVAS))
    path = OUT / "e3_support_geometry_seed10_task_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)
    return path


def write_markdown(
    summary: list[dict[str, str]],
    corr: list[dict[str, str]],
    figure: Path,
    sheet: Path,
) -> Path:
    path = OUT / "e3_support_geometry_summary.md"
    lines = [
        "# E3 Support Geometry Ablation",
        "",
        "Scope: cat_crown, tshirt_star, and backpack_remove_toy_charm; seeds 10/11/12.",
        "This diagnostic uses saved support/debug masks and fixed external evaluation masks. Attention/clean/velocity rows are support-map diagnostics, not runnable method rows.",
        "",
        f"Figure 4 panel: `{figure}`",
        f"Seed10 task sheet: `{sheet}`",
        "",
        "| Variant | n | IoU up | Precision up | Recall up | Area | Outside L1 down | Edit score | Role |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        outside_l1 = row["outside_mask_l1_mean"] or "-"
        edit_score = row["edit_score_mean"] or "-"
        lines.append(
            f"| {row['display_name']} | {row['n']} | {row['support_iou_mean']} | "
            f"{row['support_precision_mean']} | {row['support_recall_mean']} | "
            f"{row['support_area_ratio_mean']} | {outside_l1} | "
            f"{edit_score} | {row['role']} |"
        )
    lines.extend(
        [
            "",
            "## Support-Quality Correlation",
            "",
            "Correlation is computed only over runnable rows with downstream outputs: generic support and operation-conditioned support.",
            "",
            "| Support metric | Downstream metric | n | Pearson r | Spearman r |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in corr:
        if row["downstream_metric"] in {"outside_mask_l1", "source_ssim_luma", "edit_score"}:
            lines.append(
                f"| {row['support_metric']} | {row['downstream_metric']} | "
                f"{row['n']} | {row['pearson_r']} | {row['spearman_r']} |"
            )
    lines.extend(
        [
            "",
            "Interpretation: E3 treats the support/mask as an explicit experimental object. Operation-conditioned support should be discussed as geometry estimation rather than as a hidden implementation detail. Downstream metrics are attached only to runnable rows with saved outputs; diagnostic heatmaps are evaluated for support quality only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = make_rows()
    write_csv(OUT / "e3_support_geometry_mask_metrics.csv", rows, list(rows[0]))
    summary = summarize(rows)
    write_csv(OUT / "e3_support_geometry_summary.csv", summary, list(summary[0]))
    task_summary = summarize_by_task(rows)
    write_csv(OUT / "e3_support_geometry_by_task_summary.csv", task_summary, list(task_summary[0]))
    corr = correlation_rows(rows)
    write_csv(OUT / "e3_support_geometry_correlation.csv", corr, list(corr[0]))
    figure = make_figure_panel("tshirt_star", "10")
    sheet = make_seed10_sheet()
    md = write_markdown(summary, corr, figure, sheet)
    print(f"wrote {OUT / 'e3_support_geometry_mask_metrics.csv'}")
    print(f"wrote {OUT / 'e3_support_geometry_summary.csv'}")
    print(f"wrote {OUT / 'e3_support_geometry_by_task_summary.csv'}")
    print(f"wrote {OUT / 'e3_support_geometry_correlation.csv'}")
    print(f"wrote {figure}")
    print(f"wrote {sheet}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
