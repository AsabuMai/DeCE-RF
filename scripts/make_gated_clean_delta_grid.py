#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TASKS = [
    ("laptop_remove_sticker", "laptop"),
    ("fridge_remove_yellow_magnet", "fridge yellow"),
    ("fridge_remove_peach_magnet", "fridge peach"),
    ("whiteboard_remove_yellow_letter", "whiteboard"),
    ("backpack_remove_toy_charm", "backpack"),
    ("dog_remove_tennis_ball", "dog"),
]

DEFAULT_GATED_METHOD = "support_v3_controller_rmsgap_completion_clean_delta_gated_highconf"

BASE_COLUMNS = [
    ("source", "source"),
    ("support_v3_controller_rmsgap", "M0 default"),
    ("support_v3_controller_rmsgap_completion_clean_delta", "M1 clean_delta"),
]

RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_reliability(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["task"]: float(row["R"]) for row in csv.DictReader(handle)}


def load_gate(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["task"]): row for row in data.get("per_task_gate", [])}


def source_path(root: Path, task: str, seed: int) -> Path | None:
    meta_path = root / task / "support_v3_controller_rmsgap" / f"seed_{seed}" / "metadata.json"
    if not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    image = metadata.get("image")
    return Path(image) if image else None


def result_path(root: Path, task: str, method: str, seed: int) -> Path | None:
    if method == "source":
        return source_path(root, task, seed)
    return root / task / method / f"seed_{seed}" / "result.png"


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), RESAMPLE_LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def missing_cell(size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    label = "missing"
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.text(((size - (bbox[2] - bbox[0])) // 2, (size - (bbox[3] - bbox[1])) // 2), label, fill=(90, 90, 90), font=font)
    return canvas


def draw_grid(args: argparse.Namespace) -> None:
    reliability = load_reliability(args.reliability_csv)
    gates = load_gate(args.protocol)
    columns = BASE_COLUMNS + [(args.gated_method, "M2 highconf")]
    left_w = 250
    header_h = 66
    pad = 12
    row_h = args.thumb + pad
    width = left_w + len(columns) * (args.thumb + pad) + pad
    height = header_h + len(TASKS) * row_h + pad
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(18, True)
    row_font = load_font(17, True)
    small_font = load_font(14)

    for col_index, (_, label) in enumerate(columns):
        x = left_w + pad + col_index * (args.thumb + pad)
        draw.text((x, 24), label, fill=(24, 24, 24), font=header_font)

    for row_index, (task, label) in enumerate(TASKS):
        y = header_h + row_index * row_h
        gate = gates.get(task, {})
        factor = float(gate.get("gate_factor", 0.0))
        scale = float(gate.get("completion_clean_delta_scale", 0.0))
        draw.text((pad, y + 16), label, fill=(24, 24, 24), font=row_font)
        draw.text((pad, y + 42), f"R={reliability.get(task, 0.0):.2f} gate={factor:.1f} scale={scale:.3f}", fill=(70, 70, 70), font=small_font)
        for col_index, (method, _) in enumerate(columns):
            path = result_path(args.root, task, method, args.seed)
            cell = fit_image(path, args.thumb) if path is not None and path.exists() else missing_cell(args.thumb)
            x = left_w + pad + col_index * (args.thumb + pad)
            canvas.paste(cell, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the M0/M1/M2 gated clean-delta grid.")
    parser.add_argument("--root", type=Path, default=Path("outputs/pretty_matrix"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--thumb", type=int, default=230)
    parser.add_argument("--reliability-csv", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability/completion_prior_reliability_seed10.csv"))
    parser.add_argument("--gated-method", default=DEFAULT_GATED_METHOD)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_highconf_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/support_v3_2026-05-11/visual_gates/removal_completion_clean_delta_gated_highconf_seed10_grid.png"))
    args = parser.parse_args()
    draw_grid(args)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
