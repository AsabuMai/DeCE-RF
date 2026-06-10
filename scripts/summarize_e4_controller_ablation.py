from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EXP = Path("experiments/support_v3_2026-06-02")
OUT = EXP / "e4_controller_ablation"
TASKS = ["cat_crown", "tshirt_star", "pillow_vertical_fabric_strip"]
SEEDS = ["10", "11", "12"]
BASE_METHODS = ["support_v3_fixed", "support_v3_controller_rmsgap"]
LEVELS = ["0.50", "0.75", "1.00", "1.25", "1.50", "2.00"]


DISPLAY = {
    "support_v3_fixed": "Fixed DeCE displacement",
    "support_v3_controller_rmsgap": "DeCE-RF",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def avg(rows: list[dict[str, str]], key: str) -> float | None:
    values = []
    for row in rows:
        try:
            if row.get(key, "") != "":
                values.append(float(row[key]))
        except ValueError:
            pass
    return sum(values) / len(values) if values else None


def edit_proxy(rows: list[dict[str, str]]) -> float | None:
    return avg(rows, "inside_mask_l1")


def fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def parse_stress_method(method: str) -> tuple[str, str]:
    if method.endswith("_e4x050"):
        return method[: -len("_e4x050")], "0.50"
    if method.endswith("_e4x075"):
        return method[: -len("_e4x075")], "0.75"
    if method.endswith("_e4x125"):
        return method[: -len("_e4x125")], "1.25"
    if method.endswith("_e4x150"):
        return method[: -len("_e4x150")], "1.50"
    if method.endswith("_e4x200"):
        return method[: -len("_e4x200")], "2.00"
    return method, "1.00"


def summarize_base(metrics: list[dict[str, str]]) -> list[dict[str, str]]:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        by_method[row["method"]].append(row)
    out = []
    for method in BASE_METHODS:
        group = by_method[method]
        out.append(
            {
                "method": method,
                "display_name": DISPLAY[method],
                "n": str(len(group)),
                "outside_mask_l1_mean": fmt(avg(group, "outside_mask_l1")),
                "inside_mask_l1_mean": fmt(avg(group, "inside_mask_l1")),
                "source_ssim_luma_mean": fmt(avg(group, "source_ssim_luma")),
                "local_edit_l1_mean": fmt(edit_proxy(group)),
                "edit_score_mean": fmt(edit_proxy(group)),
                "claim_boundary": "component ablation; not an external baseline",
            }
        )
    return out


def summarize_stress(metrics: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        base, level = parse_stress_method(row["method"])
        if base in BASE_METHODS:
            by_key[(base, level)].append(row)
    out = []
    for base in BASE_METHODS:
        for level in LEVELS:
            group = by_key[(base, level)]
            out.append(
                {
                    "method": base,
                    "display_name": DISPLAY[base],
                    "edit_strength_multiplier": level,
                    "n": str(len(group)),
                    "outside_mask_l1_mean": fmt(avg(group, "outside_mask_l1")),
                    "inside_mask_l1_mean": fmt(avg(group, "inside_mask_l1")),
                    "source_ssim_luma_mean": fmt(avg(group, "source_ssim_luma")),
                    "local_edit_l1_mean": fmt(edit_proxy(group)),
                    "edit_score_mean": fmt(edit_proxy(group)),
                }
            )
    return out


def load_stats(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def stat_avg(stats: list[dict], key: str) -> float | None:
    values = []
    for row in stats:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def trajectory_rows() -> list[dict[str, str]]:
    rows = []
    keys = [
        "adaptive_preserve_drift",
        "adaptive_preserve_weight",
        "adaptive_edit_weight",
        "adaptive_projection_norm",
        "adaptive_clean_conflict_score",
        "adaptive_preserve_clean_correction_norm",
        "adaptive_edit_progress",
        "adaptive_edit_change_rms",
    ]
    for task in TASKS:
        for method in BASE_METHODS:
            for seed in SEEDS:
                stats_path = (
                    Path("outputs")
                    / "pretty_matrix"
                    / task
                    / method
                    / f"seed_{seed}"
                    / "stats.json"
                )
                stats = load_stats(stats_path)
                row = {
                    "task": task,
                    "method": method,
                    "display_name": DISPLAY[method],
                    "seed": seed,
                    "n_steps_logged": str(len(stats)),
                    "stats_path": str(stats_path),
                }
                for key in keys:
                    row[key + "_mean"] = fmt(stat_avg(stats, key))
                    row[key + "_max"] = fmt(max((float(s.get(key)) for s in stats if isinstance(s.get(key), (int, float))), default=0.0))
                rows.append(row)
    return rows


def summarize_trajectory(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    out = []
    for method in BASE_METHODS:
        group = by_method[method]
        row = {
            "method": method,
            "display_name": DISPLAY[method],
            "n": str(len(group)),
        }
        for key in [
            "adaptive_preserve_drift_mean",
            "adaptive_preserve_weight_mean",
            "adaptive_edit_weight_mean",
            "adaptive_projection_norm_mean",
            "adaptive_preserve_clean_correction_norm_mean",
        ]:
            row[key] = fmt(avg(group, key))
        out.append(row)
    return out


def draw_plot(
    path: Path,
    title: str,
    series: dict[str, list[tuple[float, float, str]]],
    xlabel: str,
    ylabel: str,
) -> None:
    width, height = 1200, 820
    margin_l, margin_r, margin_t, margin_b = 110, 60, 80, 100
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    xs = [x for points in series.values() for x, _, _ in points]
    ys = [y for points in series.values() for _, y, _ in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1
    pad_x = (max_x - min_x) * 0.08
    pad_y = (max_y - min_y) * 0.12
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    def sx(x: float) -> int:
        return int(margin_l + (x - min_x) / (max_x - min_x) * plot_w)

    def sy(y: float) -> int:
        return int(margin_t + plot_h - (y - min_y) / (max_y - min_y) * plot_h)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 30)
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        title_font = font = small = ImageFont.load_default()
    draw.text((margin_l, 25), title, fill=(20, 20, 20), font=title_font)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill=(40, 40, 40), width=3)
    draw.line((margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h), fill=(40, 40, 40), width=3)
    draw.text((margin_l + plot_w // 2 - 120, height - 55), xlabel, fill=(20, 20, 20), font=font)
    draw.text((20, margin_t + plot_h // 2), ylabel, fill=(20, 20, 20), font=font)
    colors = {
        "Fixed DeCE displacement": (59, 130, 246),
        "DeCE-RF": (220, 38, 38),
        "preserve drift": (22, 163, 74),
        "projection norm": (147, 51, 234),
        "preserve correction": (234, 88, 12),
    }
    fallback = [(15, 118, 110), (190, 24, 93), (75, 85, 99)]
    for idx, (name, points) in enumerate(series.items()):
        points = sorted(points, key=lambda item: float(item[2]))
        color = colors.get(name, fallback[idx % len(fallback)])
        coords = [(sx(x), sy(y)) for x, y, _ in points]
        if len(coords) > 1:
            draw.line(coords, fill=color, width=4)
        for (x, y, level), (px, py) in zip(points, coords):
            draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color, outline=(0, 0, 0))
            draw.text((px + 10, py - 18), level, fill=color, font=small)
        lx = margin_l + plot_w - 360
        ly = margin_t + 25 + idx * 34
        draw.rectangle((lx, ly, lx + 24, ly + 24), fill=color)
        draw.text((lx + 34, ly - 2), name, fill=(20, 20, 20), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def make_pareto_figure(stress_summary: list[dict[str, str]]) -> Path:
    series: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for row in stress_summary:
        if row["outside_mask_l1_mean"] and row["edit_score_mean"]:
            series[row["display_name"]].append(
                (
                    float(row["outside_mask_l1_mean"]),
                    float(row["edit_score_mean"]),
                    row["edit_strength_multiplier"],
                )
            )
    path = OUT / "e4_figure5_edit_strength_pareto.png"
    draw_plot(path, "E4 edit-strength stress Pareto", series, "outside-mask L1 (lower is better)", "local edit L1 proxy (higher change)")
    return path


def make_trajectory_figure() -> Path:
    task = "tshirt_star"
    seed = "10"
    series: dict[str, list[tuple[float, float, str]]] = {}
    for key, label in [
        ("adaptive_preserve_drift", "preserve drift"),
        ("adaptive_projection_norm", "projection norm"),
        ("adaptive_preserve_clean_correction_norm", "preserve correction"),
    ]:
        stats = load_stats(Path("outputs") / "pretty_matrix" / task / "support_v3_controller_rmsgap" / f"seed_{seed}" / "stats.json")
        points = []
        for row in stats:
            step = row.get("step")
            value = row.get(key)
            if isinstance(step, (int, float)) and isinstance(value, (int, float)):
                points.append((float(step), float(value), str(int(step))))
        if points:
            series[label] = points
    path = OUT / "e4_controller_trajectory_tshirt_star_seed10.png"
    draw_plot(path, "E4 DeCE-RF controller trajectory", series, "logged step", "controller signal")
    return path


def write_markdown(
    base_summary: list[dict[str, str]],
    stress_summary: list[dict[str, str]],
    traj_summary: list[dict[str, str]],
    pareto: Path,
    trajectory: Path,
) -> Path:
    path = OUT / "e4_controller_ablation_summary.md"
    lines = [
        "# E4 Controller And Robustness Ablation",
        "",
        "Scope: cat_crown, tshirt_star, and pillow_vertical_fabric_strip.",
        "Base fixed-vs-feedback rows use seeds 10/11/12. Edit-strength stress uses seed10 across multipliers 0.50, 0.75, 1.00, 1.25, 1.50, and 2.00.",
        "",
        f"Figure 5 Pareto: `{pareto}`",
        f"Controller trajectory figure: `{trajectory}`",
        "",
        "## Base Component Table",
        "",
        "| Variant | n | Outside L1 down | Inside L1 | Source SSIM up | Local edit L1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in base_summary:
        lines.append(
            f"| {row['display_name']} | {row['n']} | {row['outside_mask_l1_mean']} | "
            f"{row['inside_mask_l1_mean']} | {row['source_ssim_luma_mean']} | {row['edit_score_mean']} |"
        )
    lines.extend(
        [
            "",
            "## Trajectory Summary",
            "",
            "| Variant | n | Preserve drift | Preserve weight | Edit weight | Projection norm | Preserve correction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in traj_summary:
        lines.append(
            f"| {row['display_name']} | {row['n']} | {row['adaptive_preserve_drift_mean']} | "
            f"{row['adaptive_preserve_weight_mean']} | {row['adaptive_edit_weight_mean']} | "
            f"{row['adaptive_projection_norm_mean']} | {row['adaptive_preserve_clean_correction_norm_mean']} |"
        )
    lines.extend(
        [
            "",
            "## Edit-Strength Stress",
            "",
            "| Variant | Multiplier | n | Outside L1 down | Local edit L1 | Source SSIM up |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stress_summary:
        lines.append(
            f"| {row['display_name']} | {row['edit_strength_multiplier']} | {row['n']} | "
            f"{row['outside_mask_l1_mean']} | {row['edit_score_mean']} | {row['source_ssim_luma_mean']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: E4 treats feedback as a stabilizer/robustness component rather than the sole source of the headline gain. Fixed DeCE displacement keeps operation-conditioned support and fixed clean-estimate edit-preserve displacement; DeCE-RF adds feedback-updated weights, projection, and preserve clean correction. The stress curve uses fixed-mask local edit L1 as an edit-pressure proxy, so it should be discussed as an edit-preserve tradeoff rather than a standalone semantic success score.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base_metrics = read_csv(EXP / "e4_controller_base_metrics.csv")
    stress_metrics = read_csv(EXP / "e4_edit_strength_metrics.csv")
    base_summary = summarize_base(base_metrics)
    stress_summary = summarize_stress(stress_metrics)
    traj_rows = trajectory_rows()
    traj_summary = summarize_trajectory(traj_rows)

    write_csv(OUT / "e4_controller_base_summary.csv", base_summary, list(base_summary[0]))
    write_csv(OUT / "e4_edit_strength_summary.csv", stress_summary, list(stress_summary[0]))
    write_csv(OUT / "e4_controller_trajectory_stats.csv", traj_rows, list(traj_rows[0]))
    write_csv(OUT / "e4_controller_trajectory_summary.csv", traj_summary, list(traj_summary[0]))
    pareto = make_pareto_figure(stress_summary)
    trajectory = make_trajectory_figure()
    md = write_markdown(base_summary, stress_summary, traj_summary, pareto, trajectory)
    print(f"wrote {OUT / 'e4_controller_base_summary.csv'}")
    print(f"wrote {OUT / 'e4_edit_strength_summary.csv'}")
    print(f"wrote {OUT / 'e4_controller_trajectory_stats.csv'}")
    print(f"wrote {OUT / 'e4_controller_trajectory_summary.csv'}")
    print(f"wrote {pareto}")
    print(f"wrote {trajectory}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
