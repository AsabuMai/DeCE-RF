#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)

TASK_LABELS = {
    "cat_crown": "Cat crown",
    "dog_sunglasses": "Dog sunglasses",
    "mug_heart": "Mug heart",
    "backpack_remove_toy_charm": "Backpack removal",
}

METHOD_LABELS = {
    "base_only": "RF recon.",
    "direct_target": "Direct target",
    "adaptive_full_generic_support": "Generic support",
    "support_v3_fixed": "Fixed DeCE",
    "support_v3_controller_rmsgap": "DeCE-RF",
}


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


def mask_overlay(source_path: Path, mask_path: Path, size: int) -> Image.Image:
    source = Image.open(source_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(source.size, RESAMPLE_LANCZOS)
    red = Image.new("RGB", source.size, (220, 40, 35))
    overlay = Image.composite(red, source, mask.point(lambda value: int(value * 0.45)))
    overlay.thumbnail((size, size), RESAMPLE_LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - overlay.width) // 2
    y = (size - overlay.height) // 2
    canvas.paste(overlay, (x, y))
    return canvas


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pick_rows(rows: list[dict[str, str]], tasks: list[str], methods: list[str], seed: str) -> dict[tuple[str, str], dict[str, str]]:
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["task"] in tasks and row["method"] in methods and row["seed"] == seed:
            selected[(row["task"], row["method"])] = row
    return selected


def make_grid(
    rows: list[dict[str, str]],
    tasks: list[str],
    methods: list[str],
    seed: str,
    output: Path,
    include_eval_mask: bool,
    thumb: int,
) -> None:
    selected = pick_rows(rows, tasks, methods, seed)
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["task"] in tasks and row["seed"] == seed:
            by_task[row["task"]].append(row)

    columns = ["Source"]
    if include_eval_mask:
        columns.append("Eval mask")
    columns.extend(METHOD_LABELS.get(method, method) for method in methods)

    left_w = 250
    header_h = 58
    row_h = thumb + 30
    pad = 12
    width = left_w + len(columns) * (thumb + pad) + pad
    height = header_h + len(tasks) * (row_h + pad) + pad
    grid = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(grid)
    header_font = load_font(20, bold=True)
    task_font = load_font(20, bold=True)
    small_font = load_font(16)

    for col_index, label in enumerate(columns):
        x = left_w + pad + col_index * (thumb + pad)
        draw.text((x, 18), label, fill=(20, 20, 20), font=header_font)

    for row_index, task in enumerate(tasks):
        y = header_h + pad + row_index * (row_h + pad)
        draw.text((pad, y + 18), TASK_LABELS.get(task, task), fill=(20, 20, 20), font=task_font)
        task_rows = by_task.get(task, [])
        if not task_rows:
            continue

        source_path = Path(task_rows[0]["source_image"])
        mask_path = Path(task_rows[0]["mask_path"])
        cells = [fit_image(source_path, thumb)]
        if include_eval_mask:
            cells.append(mask_overlay(source_path, mask_path, thumb))
        for method in methods:
            row = selected.get((task, method))
            if row is None:
                cells.append(Image.new("RGB", (thumb, thumb), (235, 235, 235)))
            else:
                cells.append(fit_image(Path(row["result_image"]), thumb))

        for col_index, cell in enumerate(cells):
            x = left_w + pad + col_index * (thumb + pad)
            grid.paste(cell, (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-style qualitative grids from fixed-mask metrics.")
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="cat_crown dog_sunglasses mug_heart backpack_remove_toy_charm")
    parser.add_argument("--methods", default="base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap")
    parser.add_argument("--seeds", default="10")
    parser.add_argument("--include-fixed", action="store_true")
    parser.add_argument("--include-eval-mask", action="store_true")
    parser.add_argument("--thumb", type=int, default=256)
    args = parser.parse_args()

    tasks = args.tasks.split()
    methods = args.methods.split()
    if args.include_fixed and "support_v3_fixed" not in methods:
        methods = [*methods[:-1], "support_v3_fixed", methods[-1]]
    rows = read_rows(args.metrics_csv)

    for seed in args.seeds.split():
        suffix = "with_fixed" if args.include_fixed else "main"
        output = args.output_dir / f"core4_{suffix}_seed{seed}_grid.png"
        make_grid(rows, tasks, methods, seed, output, args.include_eval_mask, args.thumb)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
