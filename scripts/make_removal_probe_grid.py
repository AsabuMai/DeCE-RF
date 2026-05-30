#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)

TASKS = [
    ("backpack_remove_toy_charm", "backpack toy charm"),
    ("laptop_remove_sticker", "laptop sticker"),
    ("dog_remove_tennis_ball", "dog tennis ball"),
]

VARIANTS = [
    ("support_v3_controller_rmsgap", "default"),
    ("support_v3_controller_rmsgap_cleanfill_A", "clean_fill_A"),
    ("support_v3_controller_rmsgap_cleanfill_B", "clean_fill_B"),
    ("support_v3_controller_rmsgap_cleanfill_C", "clean_fill_C"),
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
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def missing_cell(size: int, label: str = "missing") -> Image.Image:
    canvas = Image.new("RGB", (size, size), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.text(((size - (bbox[2] - bbox[0])) // 2, (size - (bbox[3] - bbox[1])) // 2), label, fill=(90, 90, 90), font=font)
    return canvas


def metadata_path(root: Path, task: str, method: str, seed: int) -> Path:
    return root / task / method / f"seed_{seed}" / "metadata.json"


def result_path(root: Path, task: str, method: str, seed: int) -> Path:
    return root / task / method / f"seed_{seed}" / "result.png"


def source_path(root: Path, task: str, method: str, seed: int) -> Path | None:
    path = metadata_path(root, task, method, seed)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    image_path = meta.get("image")
    return Path(image_path) if image_path else None


def support_overlay(root: Path, task: str, method: str, seed: int, size: int) -> Image.Image:
    src = source_path(root, task, method, seed)
    mask = root / task / method / f"seed_{seed}" / "masks" / "subject_final.png"
    if src is None or not src.exists() or not mask.exists():
        return missing_cell(size)

    source = Image.open(src).convert("RGB")
    support = Image.open(mask).convert("L").resize(source.size, RESAMPLE_LANCZOS)
    tint = Image.new("RGB", source.size, (226, 45, 39))
    blended = Image.blend(source, tint, 0.45)
    overlay = Image.composite(blended, source, support.point(lambda value: 255 if value > 20 else 0))
    overlay.thumbnail((size, size), RESAMPLE_LANCZOS)

    canvas = Image.new("RGB", (size, size), "white")
    x = (size - overlay.width) // 2
    y = (size - overlay.height) // 2
    canvas.paste(overlay, (x, y))
    return canvas


def draw_labeled_grid(rows: list[tuple[str, str]], columns: list[tuple[str, str]], cells: list[list[Image.Image]], output: Path, thumb: int) -> None:
    left_w = 260
    header_h = 64
    pad = 14
    row_h = thumb + pad
    width = left_w + len(columns) * (thumb + pad) + pad
    height = header_h + len(rows) * row_h + pad
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(19, bold=True)
    row_font = load_font(18, bold=True)

    for col_index, (_, label) in enumerate(columns):
        x = left_w + pad + col_index * (thumb + pad)
        draw.text((x, 22), label, fill=(24, 24, 24), font=header_font)

    for row_index, (_, label) in enumerate(rows):
        y = header_h + row_index * row_h
        draw.text((pad, y + 20), label, fill=(24, 24, 24), font=row_font)
        for col_index, cell in enumerate(cells[row_index]):
            x = left_w + pad + col_index * (thumb + pad)
            canvas.paste(cell, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_result_grid(root: Path, output: Path, seed: int, thumb: int) -> None:
    cells = []
    for task, _ in TASKS:
        row = []
        for method, _ in VARIANTS:
            path = result_path(root, task, method, seed)
            row.append(fit_image(path, thumb) if path.exists() else missing_cell(thumb))
        cells.append(row)
    draw_labeled_grid(TASKS, VARIANTS, cells, output, thumb)


def make_support_grid(root: Path, output: Path, seed: int, thumb: int) -> None:
    columns = [("source", "source"), ("support", "support overlay")]
    rows = []
    for task, _ in TASKS:
        src = source_path(root, task, VARIANTS[0][0], seed)
        source_cell = fit_image(src, thumb) if src is not None and src.exists() else missing_cell(thumb)
        rows.append([source_cell, support_overlay(root, task, VARIANTS[0][0], seed, thumb)])
    draw_labeled_grid(TASKS, columns, rows, output, thumb)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build seed-10 removal probe grids.")
    parser.add_argument("--root", type=Path, default=Path("outputs/pretty_matrix"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_probe_seed10"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--thumb", type=int, default=256)
    args = parser.parse_args()

    result_output = args.output_dir / f"removal_probe_seed{args.seed}_3x4_grid.png"
    support_output = args.output_dir / f"removal_probe_seed{args.seed}_support_overlay_grid.png"
    make_result_grid(args.root, result_output, args.seed, args.thumb)
    make_support_grid(args.root, support_output, args.seed, args.thumb)
    print(result_output)
    print(support_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
