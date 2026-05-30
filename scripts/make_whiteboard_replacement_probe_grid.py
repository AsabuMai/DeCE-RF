#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/workspace/rf_h_edit")
VARIANTS = [
    ("whiteboard_probe_blank", "blank"),
    ("whiteboard_probe_blue_letter_t", "blue T"),
    ("whiteboard_probe_red_letter_a", "red A"),
    ("whiteboard_probe_blue_round_magnet", "blue magnet"),
    ("whiteboard_probe_red_star_sticker", "red star"),
]
METHODS = [
    ("support_v3_controller_rmsgap", "M0"),
    ("support_v3_controller_rmsgap_replace_editor_v0", "M1 v0"),
    ("support_v3_controller_rmsgap_replace_editor_v1", "M2 v1"),
]


def load_image(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.exists():
        img = Image.new("RGB", size, (245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.text((12, 12), "missing", fill=(180, 0, 0))
        return img
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="10 11 12")
    parser.add_argument(
        "--output",
        default="experiments/support_v3_2026-05-11/visual_gates/whiteboard_replacement_probe_seeds10_11_12_grid.png",
    )
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split()]
    tile = (190, 190)
    header_h = 42
    row_label_w = 148
    gap = 8
    cols = len(seeds) * len(METHODS)
    rows = len(VARIANTS)
    width = row_label_w + cols * tile[0] + (cols + 1) * gap
    height = header_h + rows * tile[1] + (rows + 1) * gap
    grid = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for c_seed, seed in enumerate(seeds):
        for c_method, (_, method_label) in enumerate(METHODS):
            col = c_seed * len(METHODS) + c_method
            x = row_label_w + gap + col * (tile[0] + gap)
            draw.text((x + 4, 14), f"s{seed} {method_label}", fill=(20, 20, 20), font=font)

    for r, (variant, label) in enumerate(VARIANTS):
        y = header_h + gap + r * (tile[1] + gap)
        draw.text((10, y + 8), label, fill=(20, 20, 20), font=font)
        for c_seed, seed in enumerate(seeds):
            for c_method, (method, _) in enumerate(METHODS):
                col = c_seed * len(METHODS) + c_method
                x = row_label_w + gap + col * (tile[0] + gap)
                path = (
                    ROOT
                    / "outputs"
                    / "pretty_matrix"
                    / variant
                    / method
                    / f"seed_{seed}"
                    / "result.png"
                )
                grid.paste(load_image(path, tile), (x, y))
                draw.rectangle([x, y, x + tile[0] - 1, y + tile[1] - 1], outline=(220, 220, 220))

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(out)


if __name__ == "__main__":
    main()
