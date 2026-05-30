from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLOR_RGB = {
    "black": (8, 8, 8),
    "blue": (16, 70, 230),
    "green": (16, 145, 58),
    "red": (230, 18, 18),
    "yellow": (240, 205, 18),
}


def parse_box(value: str) -> tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--box must be x0,y0,x1,y1 in normalized coordinates")
    x0, y0, x1, y1 = parts
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def parse_color(value: str) -> tuple[int, int, int]:
    key = value.lower().strip()
    if key in COLOR_RGB:
        return COLOR_RGB[key]
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--color must be a known color or r,g,b")
    if max(parts) <= 1.0:
        parts = [v * 255.0 for v in parts]
    return tuple(int(max(0, min(255, round(v)))) for v in parts)


def star_points(cx: float, cy: float, rx: float, ry: float, points: int = 5) -> list[tuple[float, float]]:
    coords = []
    for idx in range(points * 2):
        radius = 1.0 if idx % 2 == 0 else 0.42
        angle = -math.pi / 2.0 + idx * math.pi / points
        coords.append((cx + math.cos(angle) * rx * radius, cy + math.sin(angle) * ry * radius))
    return coords


def heart_points(cx: float, cy: float, rx: float, ry: float, samples: int = 96) -> list[tuple[float, float]]:
    coords = []
    for idx in range(samples):
        t = 2.0 * math.pi * idx / samples
        x = 16.0 * math.sin(t) ** 3
        y = 13.0 * math.cos(t) - 5.0 * math.cos(2.0 * t) - 2.0 * math.cos(3.0 * t) - math.cos(4.0 * t)
        coords.append((cx + (x / 18.0) * rx, cy - (y / 18.0) * ry))
    return coords


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple[int, int, int, int], fill: tuple[int, int, int] | int) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = max(1.0, (x1 - x0) / 2.0)
    ry = max(1.0, (y1 - y0) / 2.0)
    if shape == "rectangle":
        draw.rounded_rectangle(box, radius=max(1, int(min(rx, ry) * 0.08)), fill=fill)
    elif shape == "ellipse":
        draw.ellipse(box, fill=fill)
    elif shape == "heart":
        draw.polygon(heart_points(cx, cy, rx, ry), fill=fill)
    elif shape == "star":
        draw.polygon(star_points(cx, cy, rx, ry), fill=fill)
    elif shape == "leaf":
        leaf = [
            (cx, y0),
            (x1, cy - 0.10 * ry),
            (cx + 0.18 * rx, y1),
            (x0, cy + 0.10 * ry),
        ]
        draw.polygon(leaf, fill=fill)
        stem = (int(cx - 0.04 * rx), int(cy), int(cx + 0.04 * rx), int(y1 + 0.38 * ry))
        draw.rounded_rectangle(stem, radius=max(1, int(0.04 * rx)), fill=fill)
    else:
        raise ValueError(f"Unsupported shape: {shape}")


def make_overlay(image: Image.Image, mask: Image.Image, output: Path) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255.0
    color = np.array([0.0, 150.0, 255.0], dtype=np.float32)
    out = rgb * (1.0 - 0.45 * m) + color * (0.45 * m)
    Image.fromarray(out.clip(0, 255).round().astype(np.uint8)).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed decal mask and colored reference image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Reference image with decal composited.")
    parser.add_argument("--mask-output", required=True, type=Path)
    parser.add_argument("--overlay-output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--box", required=True, help="Normalized x0,y0,x1,y1 decal box.")
    parser.add_argument("--shape", choices=("rectangle", "ellipse", "heart", "star", "leaf"), default="heart")
    parser.add_argument("--color", default="red")
    parser.add_argument("--opacity", type=float, default=0.92)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = parse_box(args.box)
    pixel_box = (
        int(round(max(0.0, min(1.0, x0)) * width)),
        int(round(max(0.0, min(1.0, y0)) * height)),
        int(round(max(0.0, min(1.0, x1)) * width)),
        int(round(max(0.0, min(1.0, y1)) * height)),
    )
    color = parse_color(args.color)

    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    draw_shape(mask_draw, args.shape, pixel_box, fill=255)

    decal = Image.new("RGB", image.size, color)
    alpha = mask.point(lambda value: int(round(value * max(0.0, min(1.0, args.opacity)))))
    reference = image.copy()
    reference.paste(decal, (0, 0), alpha)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.mask_output.parent.mkdir(parents=True, exist_ok=True)
    reference.save(args.output)
    mask.save(args.mask_output)
    if args.overlay_output is not None:
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        make_overlay(image, mask, args.overlay_output)
    metadata = {
        "image": str(args.image),
        "output": str(args.output),
        "mask_output": str(args.mask_output),
        "box": args.box,
        "pixel_box": list(pixel_box),
        "shape": args.shape,
        "color": args.color,
        "opacity": float(args.opacity),
        "mask_area_gt_0_5": float((np.asarray(mask) > 127).mean()),
    }
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
