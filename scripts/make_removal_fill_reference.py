"""Build a texture-preserving fill reference for removal tasks.

Inside the removal mask the reference combines:
- low frequency from a smooth inpaint (keeps shading/lighting gradients), and
- high frequency cloned from a user-specified clean fabric patch (restores
  material grain that diffusion fills tend to paint too smooth).

Used as a weak --edit-ref-image hint plus optional final composite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


def parse_box(text: str) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in text.split(","))
    return x0, y0, x1, y1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--patch-box", required=True, help="Normalized x0,y0,x1,y1 of a clean texture patch on the same material.")
    parser.add_argument("--lowpass-kernel", type=int, default=21, help="Gaussian kernel for the shading low-pass (odd).")
    parser.add_argument("--grain-scale", type=float, default=1.0, help="Multiplier on the cloned high-frequency grain.")
    parser.add_argument("--metadata-output", type=Path, default=None)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    h, w = rgb.shape[:2]
    mask_img = Image.open(args.mask).convert("L").resize((w, h), Image.Resampling.LANCZOS)
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0

    # Smooth inpaint provides the shading base inside the mask.
    mask_u8 = (mask > 0.2).astype(np.uint8) * 255
    base = cv2.inpaint(rgb.astype(np.uint8), mask_u8, 7, cv2.INPAINT_TELEA).astype(np.float32)
    k = args.lowpass_kernel | 1
    base_low = cv2.GaussianBlur(base, (k, k), 0)

    # Clone grain (high frequency) from the patch, tiled with mirror padding.
    x0, y0, x1, y1 = parse_box(args.patch_box)
    px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    patch = rgb[py0:py1, px0:px1]
    patch_low = cv2.GaussianBlur(patch, (k, k), 0)
    grain = patch - patch_low
    reps_y = h // max(1, grain.shape[0]) + 2
    reps_x = w // max(1, grain.shape[1]) + 2
    # Mirror-tile to avoid visible seams in the repeated grain.
    tile_y = np.concatenate([grain, grain[::-1]], axis=0)
    tile_xy = np.concatenate([tile_y, tile_y[:, ::-1]], axis=1)
    tiled = np.tile(tile_xy, ((reps_y // 2) + 1, (reps_x // 2) + 1, 1))[:h, :w]

    filled = base_low + args.grain_scale * tiled
    alpha = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.5)),
        dtype=np.float32,
    )[..., None] / 255.0
    out = rgb * (1.0 - alpha) + filled * alpha
    Image.fromarray(out.clip(0, 255).round().astype(np.uint8), mode="RGB").save(args.output)

    if args.metadata_output:
        args.metadata_output.write_text(json.dumps({
            "image": str(args.image),
            "mask": str(args.mask),
            "patch_box": args.patch_box,
            "lowpass_kernel": k,
            "grain_scale": args.grain_scale,
            "output": str(args.output),
        }, indent=2))
    print(f"[fill-ref] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
