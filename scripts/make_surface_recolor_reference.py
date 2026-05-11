from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


COLOR_RGB = {
    "black": (0.02, 0.02, 0.02),
    "blue": (0.05, 0.22, 0.9),
    "brown": (0.42, 0.22, 0.08),
    "gray": (0.5, 0.5, 0.5),
    "green": (0.05, 0.55, 0.18),
    "grey": (0.5, 0.5, 0.5),
    "orange": (0.95, 0.42, 0.05),
    "red": (0.9, 0.05, 0.04),
    "silver": (0.75, 0.75, 0.72),
    "white": (0.95, 0.95, 0.92),
    "yellow": (0.95, 0.82, 0.05),
}


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def load_mask(path: Path, size: tuple[int, int], blur: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    mask = np.asarray(image, dtype=np.float32) / 255.0
    if blur > 1:
        kernel = blur + 1 if blur % 2 == 0 else blur
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
        if float(mask.max()) > 1e-6:
            mask = mask / float(mask.max())
    return mask.clip(0.0, 1.0)


def parse_color(value: str) -> np.ndarray:
    key = value.lower().strip()
    if key in COLOR_RGB:
        return np.array(COLOR_RGB[key], dtype=np.float32)
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--target-color must be a known color or r,g,b")
    rgb = np.array(parts, dtype=np.float32)
    if float(rgb.max()) > 1.0:
        rgb = rgb / 255.0
    return rgb.clip(0.0, 1.0)


def recolor_lab_chroma(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    target_rgb: np.ndarray,
    luma_rgb: np.ndarray,
    blend: float,
) -> np.ndarray:
    alpha = (mask * max(0.0, min(1.0, float(blend))))[..., None]
    target_u8 = (target_rgb.reshape(1, 1, 3) * 255.0).round().astype(np.uint8)

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    luma_lab = cv2.cvtColor(luma_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_u8, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]

    recolored_lab = lab.copy()
    recolored_lab[..., 0] = luma_lab[..., 0]
    recolored_lab[..., 1] = (1.0 - alpha[..., 0]) * lab[..., 1] + alpha[..., 0] * target_lab[1]
    recolored_lab[..., 2] = (1.0 - alpha[..., 0]) * lab[..., 2] + alpha[..., 0] * target_lab[2]
    recolored = cv2.cvtColor(recolored_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    out = image_rgb.astype(np.float32) * (1.0 - alpha) + recolored * alpha
    return out.clip(0, 255).round().astype(np.uint8)


def recolor_hsv_hue(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    target_rgb: np.ndarray,
    luma_rgb: np.ndarray,
    blend: float,
) -> np.ndarray:
    alpha = (mask * max(0.0, min(1.0, float(blend))))[..., None]
    target_u8 = (target_rgb.reshape(1, 1, 3) * 255.0).round().astype(np.uint8)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    luma_hsv = cv2.cvtColor(luma_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(target_u8, cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]

    recolored_hsv = hsv.copy()
    recolored_hsv[..., 0] = (1.0 - alpha[..., 0]) * hsv[..., 0] + alpha[..., 0] * target_hsv[0]
    recolored_hsv[..., 1] = (1.0 - alpha[..., 0]) * hsv[..., 1] + alpha[..., 0] * target_hsv[1]
    recolored_hsv[..., 2] = luma_hsv[..., 2]
    recolored = cv2.cvtColor(recolored_hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out = image_rgb.astype(np.float32) * (1.0 - alpha) + recolored * alpha
    return out.clip(0, 255).round().astype(np.uint8)


def recolor_hsv_target(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    target_rgb: np.ndarray,
    luma_rgb: np.ndarray,
    blend: float,
) -> np.ndarray:
    alpha = (mask * max(0.0, min(1.0, float(blend))))[..., None]
    target_u8 = (target_rgb.reshape(1, 1, 3) * 255.0).round().astype(np.uint8)

    luma_hsv = cv2.cvtColor(luma_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    target_hsv = cv2.cvtColor(target_u8, cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]

    recolored_hsv = np.zeros_like(luma_hsv)
    recolored_hsv[..., 0] = target_hsv[0]
    recolored_hsv[..., 1] = target_hsv[1]
    recolored_hsv[..., 2] = luma_hsv[..., 2]
    recolored = cv2.cvtColor(recolored_hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out = image_rgb.astype(np.float32) * (1.0 - alpha) + recolored * alpha
    return out.clip(0, 255).round().astype(np.uint8)


def recolor_yuv_chroma(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    target_rgb: np.ndarray,
    luma_rgb: np.ndarray,
    blend: float,
) -> np.ndarray:
    alpha = (mask * max(0.0, min(1.0, float(blend))))[..., None]
    target_u8 = (target_rgb.reshape(1, 1, 3) * 255.0).round().astype(np.uint8)

    yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV).astype(np.float32)
    luma_yuv = cv2.cvtColor(luma_rgb, cv2.COLOR_RGB2YUV).astype(np.float32)
    target_yuv = cv2.cvtColor(target_u8, cv2.COLOR_RGB2YUV).astype(np.float32)[0, 0]

    recolored_yuv = yuv.copy()
    recolored_yuv[..., 0] = luma_yuv[..., 0]
    recolored_yuv[..., 1] = (1.0 - alpha[..., 0]) * yuv[..., 1] + alpha[..., 0] * target_yuv[1]
    recolored_yuv[..., 2] = (1.0 - alpha[..., 0]) * yuv[..., 2] + alpha[..., 0] * target_yuv[2]
    recolored = cv2.cvtColor(recolored_yuv.clip(0, 255).astype(np.uint8), cv2.COLOR_YUV2RGB).astype(np.float32)
    out = image_rgb.astype(np.float32) * (1.0 - alpha) + recolored * alpha
    return out.clip(0, 255).round().astype(np.uint8)


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray, path: Path) -> None:
    alpha = np.clip(mask[..., None], 0.0, 1.0) * 0.45
    color = np.array([0.0, 155.0, 255.0], dtype=np.float32)
    out = image_rgb.astype(np.float32) * (1.0 - alpha) + color * alpha
    Image.fromarray(out.clip(0, 255).round().astype(np.uint8)).save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a generic masked surface recolor reference. Use the output as "
            "EDIT_REF_IMAGE for ODE-internal surface appearance guidance."
        )
    )
    parser.add_argument("--image", required=True, type=Path, help="Source, ODE result, or any image to recolor.")
    parser.add_argument("--surface-mask", "--mask", dest="surface_mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-color", required=True, help="Known color name or r,g,b in 0..1/0..255 scale.")
    parser.add_argument("--luma-image", type=Path, default=None, help="Optional image whose Lab L channel is preserved.")
    parser.add_argument("--mode", choices=("hsv-hue", "hsv-target", "lab-chroma", "yuv-chroma"), default="hsv-target")
    parser.add_argument("--blend", type=float, default=0.92)
    parser.add_argument("--mask-blur", type=int, default=5)
    parser.add_argument("--overlay-output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--surface-name", default=None, help="Optional human label such as shirt, bus body, backpack.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    image = load_rgb(args.image)
    size = image.size
    luma_image = load_rgb(args.luma_image, size=size) if args.luma_image is not None else image
    mask = load_mask(args.surface_mask, size=size, blur=args.mask_blur)

    image_rgb = np.asarray(image, dtype=np.uint8)
    luma_rgb = np.asarray(luma_image, dtype=np.uint8)
    target_rgb = parse_color(args.target_color)

    if args.mode == "lab-chroma":
        out = recolor_lab_chroma(
            image_rgb=image_rgb,
            mask=mask,
            target_rgb=target_rgb,
            luma_rgb=luma_rgb,
            blend=args.blend,
        )
    elif args.mode == "hsv-hue":
        out = recolor_hsv_hue(
            image_rgb=image_rgb,
            mask=mask,
            target_rgb=target_rgb,
            luma_rgb=luma_rgb,
            blend=args.blend,
        )
    elif args.mode == "hsv-target":
        out = recolor_hsv_target(
            image_rgb=image_rgb,
            mask=mask,
            target_rgb=target_rgb,
            luma_rgb=luma_rgb,
            blend=args.blend,
        )
    elif args.mode == "yuv-chroma":
        out = recolor_yuv_chroma(
            image_rgb=image_rgb,
            mask=mask,
            target_rgb=target_rgb,
            luma_rgb=luma_rgb,
            blend=args.blend,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(args.output)

    if args.overlay_output is not None:
        args.overlay_output.parent.mkdir(parents=True, exist_ok=True)
        make_overlay(image_rgb, mask, args.overlay_output)

    metadata = {
        "image": str(args.image),
        "luma_image": None if args.luma_image is None else str(args.luma_image),
        "surface_mask": str(args.surface_mask),
        "surface_name": args.surface_name,
        "output": str(args.output),
        "target_color": args.target_color,
        "target_rgb": [float(v) for v in target_rgb.tolist()],
        "mode": args.mode,
        "blend": float(args.blend),
        "mask_blur": int(args.mask_blur),
        "mask_area_gt_0_5": float((mask > 0.5).mean()),
        "purpose": "surface_recolor_reference_for_ode_guidance",
    }
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
