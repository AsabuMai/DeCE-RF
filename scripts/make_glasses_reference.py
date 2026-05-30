#!/usr/bin/env python3
"""Generate a synthetic glasses edit reference from semantic support metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def _scale_box(box: list[float], scale: int) -> list[int]:
    return [int(round(v * scale)) for v in box]


def _expand_box(box: list[float], image_size: tuple[int, int], expand_x: float, expand_y: float) -> list[int]:
    width, height = image_size
    x0, y0, x1, y1 = [float(v) for v in box]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = (x1 - x0) * (1.0 + expand_x)
    bh = (y1 - y0) * (1.0 + expand_y)
    return [
        max(0, int(round(cx - 0.5 * bw))),
        max(0, int(round(cy - 0.5 * bh))),
        min(width, int(round(cx + 0.5 * bw))),
        min(height, int(round(cy + 0.5 * bh))),
    ]


def _draw_glasses(
    draw: ImageDraw.ImageDraw,
    mask_draw: ImageDraw.ImageDraw,
    lens_boxes: list[list[int]],
    bridge_box: list[int],
    scale: int,
) -> None:
    black = (8, 9, 10, 255)
    rim = (0, 0, 0, 255)
    highlight = (54, 74, 88, 120)
    mask_white = 255

    scaled_lenses = [_scale_box(box, scale) for box in lens_boxes]
    for box in scaled_lenses:
        radius = max(3, (box[3] - box[1]) // 3)
        draw.rounded_rectangle(box, radius=radius, fill=black, outline=rim, width=max(2, scale))
        mask_draw.rounded_rectangle(box, radius=radius, fill=mask_white)
        hx0 = box[0] + int(0.18 * (box[2] - box[0]))
        hy0 = box[1] + int(0.16 * (box[3] - box[1]))
        hx1 = box[0] + int(0.46 * (box[2] - box[0]))
        hy1 = box[1] + int(0.34 * (box[3] - box[1]))
        draw.ellipse([hx0, hy0, hx1, hy1], fill=highlight)

    if len(scaled_lenses) == 2:
        left, right = sorted(scaled_lenses, key=lambda item: item[0])
        y_mid = int(round((left[1] + left[3] + right[1] + right[3]) / 4.0))
        bridge_h = max(2 * scale, int(0.12 * min(left[3] - left[1], right[3] - right[1])))
        bridge = [
            left[2] - max(scale, int(0.08 * (left[2] - left[0]))),
            y_mid - bridge_h // 2,
            right[0] + max(scale, int(0.08 * (right[2] - right[0]))),
            y_mid + bridge_h // 2,
        ]
    else:
        bridge = _scale_box(bridge_box, scale)
    draw.rounded_rectangle(bridge, radius=max(1, bridge[3] - bridge[1]), fill=rim)
    mask_draw.rounded_rectangle(bridge, radius=max(1, bridge[3] - bridge[1]), fill=mask_white)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--semantic-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-output", required=True)
    parser.add_argument("--overlay-output", default=None)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--lens-expand-x", type=float, default=0.28)
    parser.add_argument("--lens-expand-y", type=float, default=0.34)
    parser.add_argument("--mask-blur", type=float, default=1.2)
    parser.add_argument("--max-image-size", type=int, default=512)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    if max(image.size) > args.max_image_size:
        resize_scale = args.max_image_size / float(max(image.size))
        resized_w = max(16, int(round(image.width * resize_scale)))
        resized_h = max(16, int(round(image.height * resize_scale)))
        image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    metadata = json.loads(Path(args.semantic_metadata).read_text())
    lens_boxes = metadata.get("front_glasses_lens_boxes_xyxy") or []
    bridge_box = metadata.get("front_glasses_bridge_box_xyxy") or []
    if len(lens_boxes) != 2:
        raise ValueError("semantic metadata must contain two front_glasses_lens_boxes_xyxy entries")
    if not bridge_box:
        bridge_box = metadata.get("support_box_xyxy") or []
    if len(bridge_box) != 4:
        raise ValueError("semantic metadata must contain a bridge or support box")

    expanded_lenses = [_expand_box(box, image.size, args.lens_expand_x, args.lens_expand_y) for box in lens_boxes]
    scale = 4
    canvas = image.convert("RGBA").resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    mask = Image.new("L", canvas.size, 0)
    draw = ImageDraw.Draw(canvas, "RGBA")
    mask_draw = ImageDraw.Draw(mask)
    _draw_glasses(draw, mask_draw, expanded_lenses, bridge_box, scale)
    if args.mask_blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=args.mask_blur * scale))

    out = canvas.resize(image.size, Image.Resampling.LANCZOS).convert("RGB")
    out_mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    out_mask.save(args.mask_output)

    if args.overlay_output:
        overlay = image.convert("RGBA")
        red = Image.new("RGBA", image.size, (255, 32, 32, 0))
        red.putalpha(out_mask.point(lambda v: int(v * 0.45)))
        Image.alpha_composite(overlay, red).convert("RGB").save(args.overlay_output)

    if args.metadata_output:
        payload = {
            "image": args.image,
            "semantic_metadata": args.semantic_metadata,
            "output": args.output,
            "mask_output": args.mask_output,
            "lens_expand_x": args.lens_expand_x,
            "lens_expand_y": args.lens_expand_y,
            "mask_blur": args.mask_blur,
            "max_image_size": args.max_image_size,
            "image_size": list(image.size),
            "source_lens_boxes_xyxy": lens_boxes,
            "expanded_lens_boxes_xyxy": expanded_lenses,
            "bridge_box_xyxy": bridge_box,
            "front_glasses_auto_eye_used": metadata.get("front_glasses_auto_eye_used"),
        }
        Path(args.metadata_output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
