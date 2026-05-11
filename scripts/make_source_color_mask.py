from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLOR_RGB = {
    "yellow": (0.95, 0.82, 0.05),
    "blue": (0.05, 0.22, 0.9),
    "red": (0.9, 0.05, 0.04),
    "green": (0.05, 0.55, 0.18),
    "black": (0.02, 0.02, 0.02),
    "white": (0.95, 0.95, 0.92),
}


def load_gray(path: str, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(mask, dtype=np.float32) / 255.0


def keep_components(mask: np.ndarray, keep: int, min_area: int) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))
    components.sort(reverse=True)
    out = np.zeros_like(mask, dtype=np.float32)
    for _, label in components[:keep]:
        out[labels == label] = mask[labels == label]
    return out


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    if binary.max() == 0:
        return mask
    h, w = binary.shape
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 1)
    holes = (flood == 0).astype(np.float32)
    return np.maximum(mask, holes).astype(np.float32)


def make_overlay(image: Image.Image, mask: np.ndarray, output: Path) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.array([0.0, 160.0, 255.0], dtype=np.float32)
    alpha = np.clip(mask[..., None], 0.0, 1.0) * 0.55
    overlay = (rgb * (1.0 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a source-color paint mask from input pixels.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overlay-output", type=Path, default=None)
    parser.add_argument("--support-mask", type=Path, default=None)
    parser.add_argument("--box", type=str, default=None, help="Optional normalized x0,y0,x1,y1 crop gate.")
    parser.add_argument("--source-color", default="yellow")
    parser.add_argument("--hue-threshold", type=float, default=0.10)
    parser.add_argument("--lab-threshold", type=float, default=0.42)
    parser.add_argument("--min-saturation", type=float, default=0.22)
    parser.add_argument("--min-value", type=float, default=0.18)
    parser.add_argument("--max-value", type=float, default=0.98)
    parser.add_argument("--mask-threshold", type=float, default=0.35)
    parser.add_argument("--keep-components", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--fill-holes", action="store_true", default=False)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--metadata-output", type=Path, default=None)
    args = parser.parse_args()

    key = args.source_color.lower().strip()
    if key not in COLOR_RGB:
        raise ValueError(f"Unsupported source color {args.source_color!r}; known={sorted(COLOR_RGB)}")

    image = Image.open(args.image).convert("RGB")
    rgb_u8 = np.asarray(image, dtype=np.uint8)
    rgb = rgb_u8.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[..., 0] / 180.0
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0

    target_u8 = np.array(COLOR_RGB[key], dtype=np.float32).reshape(1, 1, 3)
    target_rgb_u8 = (target_u8 * 255.0).round().astype(np.uint8)
    target_hsv = cv2.cvtColor(target_rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]
    target_hue = float(target_hsv[0] / 180.0)
    hue_dist = np.minimum(np.abs(hue - target_hue), 1.0 - np.abs(hue - target_hue))
    hue_weight = 1.0 / (1.0 + np.exp((hue_dist - args.hue_threshold) / 0.025))

    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    target_lab = cv2.cvtColor(target_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0] / 255.0
    lab_dist = np.linalg.norm(lab - target_lab.reshape(1, 1, 3), axis=2)
    lab_weight = 1.0 / (1.0 + np.exp((lab_dist - args.lab_threshold) / 0.055))

    chroma_gate = (sat >= args.min_saturation) & (val >= args.min_value) & (val <= args.max_value)
    mask = np.minimum(hue_weight, lab_weight).astype(np.float32)
    mask *= chroma_gate.astype(np.float32)

    if args.support_mask is not None:
        support = load_gray(str(args.support_mask), image.size)
        mask *= support

    if args.box is not None:
        parts = [float(v.strip()) for v in args.box.split(",")]
        if len(parts) != 4:
            raise ValueError("--box must be x0,y0,x1,y1")
        x0, y0, x1, y1 = parts
        width, height = image.size
        ix0 = max(0, min(width, int(round(min(x0, x1) * width))))
        ix1 = max(0, min(width, int(round(max(x0, x1) * width))))
        iy0 = max(0, min(height, int(round(min(y0, y1) * height))))
        iy1 = max(0, min(height, int(round(max(y0, y1) * height))))
        box_mask = np.zeros_like(mask, dtype=np.float32)
        box_mask[iy0:iy1, ix0:ix1] = 1.0
        mask *= box_mask

    if args.mask_threshold > 0.0:
        mask = np.where(mask >= float(args.mask_threshold), mask, 0.0).astype(np.float32)

    if args.open_kernel > 1:
        kernel = np.ones((args.open_kernel, args.open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if args.close_kernel > 1:
        kernel = np.ones((args.close_kernel, args.close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if args.keep_components > 0:
        mask = keep_components(mask, args.keep_components, args.min_area)
    if args.fill_holes:
        mask = fill_binary_holes(mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L").save(args.output)
    if args.overlay_output is not None:
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        make_overlay(image, mask, args.overlay_output)
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(
                {
                    "image": str(args.image),
                    "support_mask": None if args.support_mask is None else str(args.support_mask),
                    "box": args.box,
                    "source_color": key,
                    "target_hue": target_hue,
                    "mask_mean": float(mask.mean()),
                    "mask_area_gt_0_5": float((mask > 0.5).mean()),
                    "hue_threshold": args.hue_threshold,
                    "lab_threshold": args.lab_threshold,
                    "min_saturation": args.min_saturation,
                    "min_value": args.min_value,
                    "max_value": args.max_value,
                    "mask_threshold": args.mask_threshold,
                    "fill_holes": bool(args.fill_holes),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
