#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TASKS = [
    "laptop_remove_sticker",
    "fridge_remove_yellow_magnet",
    "fridge_remove_peach_magnet",
    "whiteboard_remove_yellow_letter",
    "backpack_remove_toy_charm",
    "dog_remove_tennis_ball",
]

LABELS = {
    "laptop_remove_sticker": "laptop",
    "fridge_remove_yellow_magnet": "fridge-y",
    "fridge_remove_peach_magnet": "fridge-p",
    "whiteboard_remove_yellow_letter": "whiteboard",
    "backpack_remove_toy_charm": "backpack",
    "dog_remove_tennis_ball": "dog",
}

METHOD_DEFAULT = "support_v3_controller_rmsgap"
METHOD_CLEAN = "support_v3_controller_rmsgap_completion_clean_delta"
DEFAULT_METHOD_GATED = "support_v3_controller_rmsgap_completion_clean_delta_gated_highconf"
METHOD_TELEA = "same_support_inpaint_telea"


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(mask, dtype=np.uint8) > 0


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        return float(np.mean(np.abs(a - b)))
    return float(np.mean(np.abs(a[mask] - b[mask])))


def load_reliability(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["task"]] = {
                "R": float(row["R"]),
                "R_boundary": float(row["R_boundary"]),
                "R_agreement": float(row["R_agreement"]),
                "R_host": float(row["R_host"]),
            }
    return out


def load_gates(protocol: Path) -> dict[tuple[str, int], dict[str, float]]:
    data = json.loads(protocol.read_text(encoding="utf-8"))
    out = {}
    for row in data.get("per_task_gate", []):
        out[(str(row["task"]), int(row["seed"]))] = {
            "gate_factor": float(row["gate_factor"]),
            "completion_clean_delta_scale": float(row["completion_clean_delta_scale"]),
        }
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def draw_scatter(rows: list[dict[str, object]], output: Path) -> None:
    width, height = 1280, 620
    margin_l, margin_r, margin_t, margin_b = 82, 36, 70, 78
    gutter = 72
    plot_w = (width - margin_l - margin_r - gutter) // 2
    plot_h = height - margin_t - margin_b
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22, True)
    font = load_font(15)
    small = load_font(12)

    xs = [float(row["R"]) for row in rows]
    y_groups = [
        ("Ungated clean-delta gain", "Gain"),
        ("Gated clean-delta gain", "GatedGain"),
    ]
    x_min, x_max = 0.0, max(0.8, max(xs) + 0.05)

    draw.text((margin_l, 18), "Reliability score vs completion gain", fill=(20, 20, 20), font=title_font)

    colors = {
        1.0: (35, 112, 181),
        0.5: (220, 145, 35),
        0.0: (150, 70, 70),
    }
    for panel_index, (panel_title, value_key) in enumerate(y_groups):
        x0 = margin_l + panel_index * (plot_w + gutter)
        ys = [float(row[value_key]) for row in rows]
        y_min = min(-0.02, min(ys) - 0.02)
        y_max = max(0.02, max(ys) + 0.02)

        def px(x: float) -> int:
            return int(x0 + (x - x_min) / max(x_max - x_min, 1e-6) * plot_w)

        def py(y: float) -> int:
            return int(margin_t + (y_max - y) / max(y_max - y_min, 1e-6) * plot_h)

        draw.text((x0, margin_t - 34), panel_title, fill=(30, 30, 30), font=font)
        draw.line((x0, margin_t + plot_h, x0 + plot_w, margin_t + plot_h), fill=(50, 50, 50), width=2)
        draw.line((x0, margin_t, x0, margin_t + plot_h), fill=(50, 50, 50), width=2)
        if y_min < 0 < y_max:
            y0 = py(0.0)
            draw.line((x0, y0, x0 + plot_w, y0), fill=(190, 190, 190), width=1)
        for tick in np.linspace(0.0, x_max, 5):
            x = px(float(tick))
            draw.line((x, margin_t + plot_h, x, margin_t + plot_h + 5), fill=(50, 50, 50), width=1)
            draw.text((x - 12, margin_t + plot_h + 10), f"{tick:.2f}", fill=(60, 60, 60), font=small)
        for tick in np.linspace(y_min, y_max, 5):
            y = py(float(tick))
            draw.line((x0 - 5, y, x0, y), fill=(50, 50, 50), width=1)
            draw.text((x0 - 58, y - 7), f"{tick:.2f}", fill=(60, 60, 60), font=small)

        for row in rows:
            x = px(float(row["R"]))
            y = py(float(row[value_key]))
            gate = float(row["gate_factor"])
            color = colors.get(gate, (80, 80, 80))
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(20, 20, 20))
            draw.text((x + 8, y - 8), f"{LABELS[str(row['task'])]} s{row['seed']}", fill=(30, 30, 30), font=small)
        draw.text((x0 + plot_w // 2 - 28, height - 34), "R_total", fill=(30, 30, 30), font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare completion prior reliability with clean-delta gain.")
    parser.add_argument("--root", type=Path, default=Path("outputs/pretty_matrix"))
    parser.add_argument("--seeds", default="10 11 12")
    parser.add_argument("--tasks", default=" ".join(TASKS))
    parser.add_argument("--reliability-dir", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability"))
    parser.add_argument("--gated-method", default=DEFAULT_METHOD_GATED)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_highconf_seeds10_11_12_protocol.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability/completion_reliability_vs_gain_highconf_seed10_11_12.csv"))
    parser.add_argument("--output-scatter", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability/completion_reliability_vs_gain_highconf_scatter.png"))
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.replace(",", " ").split() if item]
    tasks = [item for item in args.tasks.replace(",", " ").split() if item]
    gates = load_gates(args.protocol)
    rows: list[dict[str, object]] = []

    for seed in seeds:
        reliability = load_reliability(args.reliability_dir / f"completion_prior_reliability_seed{seed}.csv")
        for task in tasks:
            telea_path = args.root / task / METHOD_TELEA / f"seed_{seed}" / "result.png"
            mask_path = args.root / task / METHOD_TELEA / f"seed_{seed}" / "masks" / "same_support_inpaint_mask.png"
            telea = load_rgb(telea_path)
            mask = load_mask(mask_path, (telea.shape[1], telea.shape[0]))
            default = load_rgb(args.root / task / METHOD_DEFAULT / f"seed_{seed}" / "result.png")
            clean = load_rgb(args.root / task / METHOD_CLEAN / f"seed_{seed}" / "result.png")
            gated = load_rgb(args.root / task / args.gated_method / f"seed_{seed}" / "result.png")
            default_telea = masked_mae(default, telea, mask)
            clean_telea = masked_mae(clean, telea, mask)
            gated_telea = masked_mae(gated, telea, mask)
            gate = gates.get((task, seed), gates.get((task, 10), {"gate_factor": 0.0, "completion_clean_delta_scale": 0.0}))
            rel = reliability[task]
            rows.append(
                {
                    "task": task,
                    "seed": seed,
                    "R": rel["R"],
                    "R_boundary": rel["R_boundary"],
                    "R_agreement": rel["R_agreement"],
                    "R_host": rel["R_host"],
                    "gate_factor": gate["gate_factor"],
                    "completion_clean_delta_scale": gate["completion_clean_delta_scale"],
                    "mae_default_to_telea": default_telea,
                    "mae_clean_delta_to_telea": clean_telea,
                    "mae_gated_to_telea": gated_telea,
                    "Gain": default_telea - clean_telea,
                    "GatedGain": default_telea - gated_telea,
                    "mae_gated_default": masked_mae(gated, default, mask),
                    "mae_gated_ungated": masked_mae(gated, clean, mask),
                    "mae_clean_delta_default": masked_mae(clean, default, mask),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    draw_scatter(rows, args.output_scatter)

    print(args.output_csv)
    print(args.output_scatter)
    print(f"pearson_R_gain={pearson([float(r['R']) for r in rows], [float(r['Gain']) for r in rows]):.4f}")
    print(f"pearson_R_gated_gain={pearson([float(r['R']) for r in rows], [float(r['GatedGain']) for r in rows]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
