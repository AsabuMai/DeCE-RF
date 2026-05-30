#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)

TASKS = [
    ("backpack_remove_toy_charm", "backpack charm"),
    ("laptop_remove_sticker", "laptop sticker"),
    ("fridge_remove_yellow_magnet", "fridge yellow"),
    ("fridge_remove_peach_magnet", "fridge peach"),
    ("whiteboard_remove_yellow_letter", "whiteboard letter"),
    ("dog_remove_tennis_ball", "dog ball"),
]

COLUMNS = [
    ("source", "source"),
    ("support_v3_controller_rmsgap", "DeCE default"),
    ("same_support_inpaint_telea", "Telea ref"),
    ("support_v3_controller_rmsgap_completion_ref", "DeCE + ref"),
]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), RESAMPLE_LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def missing_cell(size: int, label: str = "missing") -> Image.Image:
    canvas = Image.new("RGB", (size, size), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.text(((size - (bbox[2] - bbox[0])) // 2, (size - (bbox[3] - bbox[1])) // 2), label, fill=(90, 90, 90), font=font)
    return canvas


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


def draw_grid(root: Path, output: Path, seed: int, thumb: int) -> None:
    left_w = 250
    header_h = 66
    pad = 12
    row_h = thumb + pad
    width = left_w + len(COLUMNS) * (thumb + pad) + pad
    height = header_h + len(TASKS) * row_h + pad
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(18, bold=True)
    row_font = load_font(17, bold=True)

    for col_index, (_, label) in enumerate(COLUMNS):
        x = left_w + pad + col_index * (thumb + pad)
        draw.text((x, 24), label, fill=(24, 24, 24), font=header_font)

    for row_index, (task, label) in enumerate(TASKS):
        y = header_h + row_index * row_h
        draw.text((pad, y + 20), label, fill=(24, 24, 24), font=row_font)
        for col_index, (method, _) in enumerate(COLUMNS):
            path = result_path(root, task, method, seed)
            cell = fit_image(path, thumb) if path is not None and path.exists() else missing_cell(thumb)
            x = left_w + pad + col_index * (thumb + pad)
            canvas.paste(cell, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 6-task completion-guided removal grid.")
    parser.add_argument("--root", type=Path, default=Path("outputs/pretty_matrix"))
    parser.add_argument("--output", type=Path, default=Path("experiments/support_v3_2026-05-11/visual_gates/removal_completion_ref_seed10_grid.png"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--thumb", type=int, default=256)
    args = parser.parse_args()
    draw_grid(args.root, args.output, args.seed, args.thumb)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
