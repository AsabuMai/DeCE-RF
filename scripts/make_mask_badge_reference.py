from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import cv2
except Exception:  # pragma: no cover - fallback path is exercised only without opencv.
    cv2 = None


COLOR_RGB = {
    "blue": (18, 88, 220),
    "green": (20, 150, 80),
    "red": (220, 28, 28),
    "yellow": (235, 195, 30),
}


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


def tight_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("semantic mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def estimate_background_color(image: Image.Image, semantic_mask: Image.Image) -> tuple[int, int, int]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hard = np.asarray(semantic_mask) > 127
    dilated = np.asarray(semantic_mask.filter(ImageFilter.MaxFilter(41))) > 0
    ring = dilated & ~hard
    if ring.sum() < 32:
        ring = ~hard
    samples = rgb[ring]
    if len(samples) == 0:
        samples = rgb.reshape(-1, 3)
    return tuple(int(v) for v in np.median(samples, axis=0))


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


def draw_replacement_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple[int, int, int, int], fill: int) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = max(1.0, (x1 - x0) / 2.0)
    ry = max(1.0, (y1 - y0) / 2.0)
    if shape == "ellipse":
        draw.ellipse(box, fill=fill)
    elif shape == "star":
        draw.polygon(star_points(cx, cy, rx, ry), fill=fill)
    elif shape == "heart":
        draw.polygon(heart_points(cx, cy, rx, ry), fill=fill)
    else:
        raise ValueError(f"Unsupported replacement shape: {shape}")


def remove_source_object(
    image: Image.Image,
    semantic_mask: Image.Image,
    *,
    mode: str,
    dilate: int,
    radius: float,
) -> tuple[Image.Image, tuple[int, int, int] | None, tuple[int, int] | None]:
    if mode == "median" or cv2 is None:
        background_color = estimate_background_color(image, semantic_mask)
        filled = image.copy()
        alpha = semantic_mask.point(lambda value: 255 if value > 127 else 0)
        filled.paste(Image.new("RGB", image.size, background_color), (0, 0), alpha)
        return filled, background_color, None

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    hard = (np.asarray(semantic_mask.convert("L")) > 127).astype(np.uint8) * 255
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        hard = cv2.dilate(hard, kernel, iterations=1)
    if mode == "patch":
        x0, y0, x1, y1 = tight_box(hard)
        box_w = x1 - x0
        box_h = y1 - y0
        candidates = [
            (box_w + 24, 0),
            (box_w + 64, 0),
            (max(64, box_w // 2), 0),
            (0, box_h + 24),
            (-(box_w + 24), 0),
        ]
        height, width = hard.shape
        best_shift = None
        best_score = None
        for dx, dy in candidates:
            sx0, sy0, sx1, sy1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
            if sx0 < 0 or sy0 < 0 or sx1 > width or sy1 > height:
                continue
            source_mask_overlap = float((hard[sy0:sy1, sx0:sx1] > 0).mean())
            patch = rgb[sy0:sy1, sx0:sx1].astype(np.float32)
            gradient = float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean())
            score = source_mask_overlap * 1000.0 + gradient
            if best_score is None or score < best_score:
                best_score = score
                best_shift = (dx, dy)
        if best_shift is not None:
            dx, dy = best_shift
            yy, xx = np.where(hard > 0)
            src_x = np.clip(xx + dx, 0, width - 1)
            src_y = np.clip(yy + dy, 0, height - 1)
            filled = rgb.copy()
            filled[yy, xx] = rgb[src_y, src_x]
            feather = cv2.GaussianBlur(hard, (0, 0), sigmaX=2.5, sigmaY=2.5).astype(np.float32) / 255.0
            feather = feather[..., None]
            out = rgb.astype(np.float32) * (1.0 - feather) + filled.astype(np.float32) * feather
            return Image.fromarray(out.clip(0, 255).round().astype(np.uint8)), None, best_shift

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, hard, radius, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)), None, None


def make_overlay(image: Image.Image, mask: Image.Image, output: Path) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255.0
    color = np.array([0.0, 150.0, 255.0], dtype=np.float32)
    out = rgb * (1.0 - 0.45 * m) + color * (0.45 * m)
    Image.fromarray(out.clip(0, 255).round().astype(np.uint8)).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a badge replacement reference from an automatic semantic mask.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--semantic-mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mask-output", required=True, type=Path)
    parser.add_argument("--overlay-output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--color", default="blue")
    parser.add_argument("--badge-shape", choices=("ellipse", "semantic", "star", "heart"), default="ellipse")
    parser.add_argument("--opacity", type=float, default=0.92)
    parser.add_argument("--scale", type=float, default=0.78)
    parser.add_argument("--blur", type=float, default=0.7)
    parser.add_argument("--background-mode", choices=("patch", "inpaint", "median"), default="patch")
    parser.add_argument("--inpaint-dilate", type=int, default=9)
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--outline", action="store_true", default=False)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    semantic_mask = Image.open(args.semantic_mask).convert("L").resize(image.size)
    semantic_arr = np.asarray(semantic_mask)
    x0, y0, x1, y1 = tight_box(semantic_arr)

    width = x1 - x0
    height = y1 - y0
    scale = max(0.2, min(1.0, args.scale))
    badge_w = max(2, int(round(width * scale)))
    badge_h = max(2, int(round(min(width, height) * scale)))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bx0 = int(round(cx - badge_w / 2.0))
    by0 = int(round(cy - badge_h / 2.0))
    bx1 = int(round(cx + badge_w / 2.0))
    by1 = int(round(cy + badge_h / 2.0))

    color = parse_color(args.color)
    cleaned_reference, background_color, background_patch_shift = remove_source_object(
        image,
        semantic_mask,
        mode=args.background_mode,
        dilate=args.inpaint_dilate,
        radius=args.inpaint_radius,
    )
    badge_mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(badge_mask)
    if args.badge_shape == "semantic":
        badge_mask = semantic_mask.point(lambda value: 255 if value > 127 else 0)
    else:
        draw_replacement_shape(draw, args.badge_shape, (bx0, by0, bx1, by1), fill=255)
    if args.blur > 0:
        badge_mask = badge_mask.filter(ImageFilter.GaussianBlur(radius=args.blur))

    rim_mask = Image.new("L", image.size, 0)
    rim_draw = ImageDraw.Draw(rim_mask)
    if args.outline:
        outline_w = max(3, int(round(min(badge_w, badge_h) * 0.08)))
        inner = badge_mask.filter(ImageFilter.MinFilter(outline_w | 1))
        rim_arr = np.maximum(0, np.asarray(badge_mask, dtype=np.int16) - np.asarray(inner, dtype=np.int16))
        rim_mask = Image.fromarray(rim_arr.clip(0, 255).astype(np.uint8), mode="L")
    highlight_mask = Image.new("L", image.size, 0)
    highlight_draw = ImageDraw.Draw(highlight_mask)
    if args.badge_shape == "ellipse":
        highlight_draw.ellipse(
            (
                int(round(bx0 + badge_w * 0.22)),
                int(round(by0 + badge_h * 0.18)),
                int(round(bx0 + badge_w * 0.46)),
                int(round(by0 + badge_h * 0.36)),
            ),
            fill=90,
        )

    reference = cleaned_reference
    alpha = badge_mask.point(lambda value: int(round(value * max(0.0, min(1.0, args.opacity)))))
    reference.paste(Image.new("RGB", image.size, color), (0, 0), alpha)
    reference.paste(Image.new("RGB", image.size, (70, 10, 10)), (0, 0), rim_mask.point(lambda value: int(value * 0.55)))
    reference.paste(Image.new("RGB", image.size, (235, 245, 255)), (0, 0), highlight_mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.mask_output.parent.mkdir(parents=True, exist_ok=True)
    reference.save(args.output)
    badge_mask.save(args.mask_output)
    if args.overlay_output is not None:
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        make_overlay(image, badge_mask, args.overlay_output)

    metadata = {
        "image": str(args.image),
        "semantic_mask": str(args.semantic_mask),
        "output": str(args.output),
        "mask_output": str(args.mask_output),
        "semantic_pixel_box": [x0, y0, x1, y1],
        "badge_pixel_box": [bx0, by0, bx1, by1],
        "color": args.color,
        "badge_shape": args.badge_shape,
        "background_mode": args.background_mode,
        "semantic_fill_color": list(background_color) if background_color is not None else None,
        "background_patch_shift": list(background_patch_shift) if background_patch_shift is not None else None,
        "inpaint_dilate": int(args.inpaint_dilate),
        "inpaint_radius": float(args.inpaint_radius),
        "opacity": float(args.opacity),
        "scale": float(scale),
        "mask_area_gt_0_5": float((np.asarray(badge_mask) > 127).mean()),
    }
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
