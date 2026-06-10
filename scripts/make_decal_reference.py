from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


COLOR_RGB = {
    "black": (8, 8, 8),
    "blue": (16, 70, 230),
    "green": (16, 145, 58),
    "red": (230, 18, 18),
    "yellow": (240, 205, 18),
}


def _box_grid(size: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = max(1.0, (x0 + x1) / 2.0)
    cy = max(1.0, (y0 + y1) / 2.0)
    rx = max(1.0, (x1 - x0) / 2.0)
    ry = max(1.0, (y1 - y0) / 2.0)
    return (xx - cx) / rx, (yy - cy) / ry


def make_orange_shadow_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    width, height = size
    x0, y0, x1, y1 = box
    fruit_w = max(1, x1 - x0)
    fruit_h = max(1, y1 - y0)
    shadow = Image.new("L", size, 0)
    draw = ImageDraw.Draw(shadow)
    cast_shadow_box = (
        int(round(x0 + fruit_w * 0.00)),
        int(round(y1 - fruit_h * 0.16)),
        int(round(x1 + fruit_w * 0.26)),
        int(round(min(height, y1 + fruit_h * 0.13))),
    )
    contact_shadow_box = (
        int(round(x0 + fruit_w * 0.10)),
        int(round(y1 - fruit_h * 0.12)),
        int(round(x1 - fruit_w * 0.00)),
        int(round(min(height, y1 + fruit_h * 0.035))),
    )
    draw.ellipse(cast_shadow_box, fill=150)
    draw.ellipse(contact_shadow_box, fill=235)
    return shadow.filter(ImageFilter.GaussianBlur(max(1.0, fruit_h * 0.030)))


def make_orange_fruit_reference(
    image: Image.Image,
    fruit_mask: Image.Image,
    shadow_mask: Image.Image,
    box: tuple[int, int, int, int],
    opacity: float,
) -> Image.Image:
    img = np.asarray(image).astype(np.float32)
    alpha = np.asarray(fruit_mask).astype(np.float32) / 255.0
    shadow = np.asarray(shadow_mask).astype(np.float32) / 255.0
    gx, gy = _box_grid(image.size, box)
    radius = np.sqrt(gx * gx + gy * gy)
    sphere = np.clip(1.0 - radius, 0.0, 1.0)
    highlight = np.exp(-(((gx + 0.35) / 0.28) ** 2 + ((gy + 0.38) / 0.22) ** 2))
    lower_shadow = np.clip((gy + 0.10) * 0.44, 0.0, 0.38)
    contact_occlusion = np.exp(-(((gx - 0.04) / 0.72) ** 2 + ((gy - 0.92) / 0.16) ** 2))
    texture = 0.010 * np.sin((gx * 34.0) + (gy * 11.0)) + 0.007 * np.sin((gx * 17.0) - (gy * 29.0))
    base = np.array([232.0, 88.0, 16.0], dtype=np.float32)
    color = base * (0.86 + 0.22 * sphere + texture - lower_shadow - 0.18 * contact_occlusion)[..., None]
    color += np.array([34.0, 30.0, 8.0], dtype=np.float32) * highlight[..., None]
    color = np.clip(color, 0.0, 255.0)
    a = np.clip(alpha * max(0.0, min(1.0, opacity)), 0.0, 1.0)[..., None]
    out = img * (1.0 - shadow[..., None] * 0.52)
    out = out * (1.0 - a) + color * a
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def make_printed_star_reference(image: Image.Image, mask: Image.Image, color: tuple[int, int, int], opacity: float) -> Image.Image:
    img = np.asarray(image).astype(np.float32)
    alpha = np.asarray(mask).astype(np.float32) / 255.0
    luma = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]) / 255.0
    low = np.asarray(
        Image.fromarray((luma * 255.0).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(9)),
        dtype=np.float32,
    ) / 255.0
    high = np.clip(luma - low, -0.10, 0.10)
    cloth_shade = np.clip(0.82 + 0.72 * (low - 0.84) + 1.20 * high, 0.70, 1.08)
    ink = np.array(color, dtype=np.float32) * cloth_shade[..., None]
    ink[..., 0] = np.maximum(ink[..., 0], 150.0)
    ink[..., 1:] *= 0.70
    a = np.clip(alpha * max(0.0, min(1.0, opacity)), 0.0, 1.0)[..., None]
    out = img * (1.0 - a * 0.66) + ink * (a * 0.66)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


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


def draw_shape(
    draw: ImageDraw.ImageDraw,
    shape: str,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int] | int,
    slant_x: float = 0.0,
    perspective_y: float = 0.0,
) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = max(1.0, (x1 - x0) / 2.0)
    ry = max(1.0, (y1 - y0) / 2.0)
    if shape == "rectangle":
        draw.rounded_rectangle(box, radius=max(1, int(min(rx, ry) * 0.08)), fill=fill)
    elif shape in {"slanted_rectangle", "corduroy_panel", "linen_panel", "terry_panel", "quilted_panel", "waffle_panel"}:
        shift = int(round((x1 - x0) * slant_x))
        taper = int(round((x1 - x0) * perspective_y))
        points = [
            (x0 + shift + taper, y0),
            (x1 + shift - taper, y0),
            (x1 - shift + taper, y1),
            (x0 - shift - taper, y1),
        ]
        draw.polygon(points, fill=fill)
    elif shape in {"ellipse", "orange_fruit"}:
        draw.ellipse(box, fill=fill)
    elif shape == "heart":
        draw.polygon(heart_points(cx, cy, rx, ry), fill=fill)
    elif shape in {"star", "printed_star"}:
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
    elif shape == "stripes":
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        stripe_h = max(2, int(round(height / 12.0)))
        gap = max(stripe_h, int(round(height / 8.0)))
        for y in range(y0 + stripe_h, y1, gap):
            draw.rounded_rectangle(
                (x0, y, x1, min(y1, y + stripe_h)),
                radius=max(1, min(stripe_h // 2, width // 40)),
                fill=fill,
            )
    elif shape == "dots":
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        cols = max(3, min(6, int(round(width / max(1.0, height) * 3.5))))
        rows = max(3, min(5, int(round(height / max(1.0, width) * 4.5))))
        step_x = width / float(cols)
        step_y = height / float(rows)
        radius = max(3, int(round(min(step_x, step_y) * 0.28)))
        for row in range(rows):
            for col in range(cols):
                cx_dot = x0 + (col + 0.5 + (0.25 if row % 2 else 0.0)) * step_x
                cy_dot = y0 + (row + 0.5) * step_y
                if cx_dot + radius > x1:
                    continue
                dot_box = (
                    int(round(cx_dot - radius)),
                    int(round(cy_dot - radius)),
                    int(round(cx_dot + radius)),
                    int(round(cy_dot + radius)),
                )
                draw.ellipse(dot_box, fill=fill)
    elif shape == "cross":
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        bar_w = max(2, int(round(width * 0.24)))
        bar_h = max(2, int(round(height * 0.24)))
        vertical = (
            int(round(cx - bar_w / 2.0)),
            y0,
            int(round(cx + bar_w / 2.0)),
            y1,
        )
        horizontal = (
            x0,
            int(round(cy - bar_h / 2.0)),
            x1,
            int(round(cy + bar_h / 2.0)),
        )
        radius = max(1, int(round(min(bar_w, bar_h) * 0.25)))
        draw.rounded_rectangle(vertical, radius=radius, fill=fill)
        draw.rounded_rectangle(horizontal, radius=radius, fill=fill)
    else:
        raise ValueError(f"Unsupported shape: {shape}")


def make_overlay(image: Image.Image, mask: Image.Image, output: Path) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255.0
    color = np.array([0.0, 150.0, 255.0], dtype=np.float32)
    out = rgb * (1.0 - 0.45 * m) + color * (0.45 * m)
    Image.fromarray(out.clip(0, 255).round().astype(np.uint8)).save(output)


def clip_material_mask_to_local_host(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, bool]:
    """Keep bright-material panels on the local host instead of spilling onto background.

    Returns the (possibly clipped) mask and whether the bright-host clip was applied.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_arr = np.asarray(mask.convert("L"), dtype=np.float32)
    active = mask_arr > 12.0
    if not np.any(active):
        return mask, False
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    active_luma = luma[active]
    p20 = float(np.percentile(active_luma, 20))
    p80 = float(np.percentile(active_luma, 80))
    # Only apply this guard for bright same-color hosts such as white pillows.
    # The surrounding sofa/chair can be midtone, so relying on p80-p20 alone
    # lets material stripes leak outside the pillow when the panel is mostly
    # bright host. A high p80 is the reliable signal that this is a bright-host
    # material panel that should be clipped by local luminance.
    if p80 < 165.0:
        return mask, False
    threshold = max(150.0, min(220.0, 0.72 * p80))
    host = ((luma >= threshold) & active).astype(np.uint8) * 255
    host_mask = Image.fromarray(host, mode="L").filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(7))
    clipped = mask_arr * (np.asarray(host_mask, dtype=np.float32) / 255.0)
    return Image.fromarray(clipped.clip(0, 255).round().astype(np.uint8), mode="L"), True


def clip_mask_to_host_color(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Remove mask pixels whose chroma deviates from the mask-core median.

    Guards dark hosts against semantic-mask over-segmentation (e.g. SAM
    including a wedge of warm rattan next to a grey pillow corner): the host
    fabric is near-uniform in chroma, so chroma outliers inside the mask are
    background, not host."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask_arr = np.asarray(mask.convert("L"), dtype=np.float32)
    core_img = Image.fromarray((mask_arr > 180).astype(np.uint8) * 255, mode="L").filter(ImageFilter.MinFilter(15))
    core = np.asarray(core_img, dtype=np.float32) > 128
    if core.sum() < 200:
        return mask
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    cr = rgb[..., 0] - luma
    cb = rgb[..., 2] - luma
    med_cr = float(np.median(cr[core]))
    med_cb = float(np.median(cb[core]))
    mad_cr = float(np.median(np.abs(cr[core] - med_cr))) + 1.5
    mad_cb = float(np.median(np.abs(cb[core] - med_cb))) + 1.5
    dist = np.abs(cr - med_cr) / mad_cr + np.abs(cb - med_cb) / mad_cb
    keep = (dist <= 6.0).astype(np.uint8) * 255
    keep_img = (
        Image.fromarray(keep, mode="L")
        .filter(ImageFilter.MaxFilter(3))
        .filter(ImageFilter.MinFilter(7))
        .filter(ImageFilter.MaxFilter(5))
        .filter(ImageFilter.GaussianBlur(radius=1.0))
    )
    out = mask_arr * (np.asarray(keep_img, dtype=np.float32) / 255.0)
    return Image.fromarray(out.clip(0, 255).round().astype(np.uint8), mode="L")


def intersect_mask_with_host(mask: Image.Image, host_mask_path: Path, size: tuple[int, int], tight: bool = False) -> Image.Image:
    """Intersect a material mask with a semantic host mask so texture stays on
    the host. Dark hosts (no luminance clip) get slight dilation to avoid
    cutting fabric edges; bright hosts get slight erosion so the latent-space
    final-mask feather cannot halo past the host outline."""
    host = Image.open(host_mask_path).convert("L").resize(size, Image.Resampling.LANCZOS)
    if tight:
        host = host.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(radius=1.0))
    else:
        host = host.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(radius=1.2))
    out = np.asarray(mask.convert("L"), dtype=np.float32) * (np.asarray(host, dtype=np.float32) / 255.0)
    return Image.fromarray(out.clip(0, 255).round().astype(np.uint8), mode="L")


def make_material_panel_reference(
    image: Image.Image,
    mask: Image.Image,
    opacity: float,
    material: str,
    contrast: float = 1.0,
    scale: float = 1.0,
) -> Image.Image:
    """Create same-color material texture inside the mask while preserving source chroma."""
    bbox = mask.getbbox()
    if bbox is None:
        return image.copy()

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask.convert("L"), dtype=np.float32)[..., None] / 255.0
    alpha *= max(0.0, min(1.0, opacity))

    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x0, y0, x1, y1 = bbox
    panel_w = max(1.0, float(x1 - x0))
    panel_h = max(1.0, float(y1 - y0))

    if material == "corduroy":
        rib_period = max(5.0, panel_w / 13.0 * max(0.25, scale))
        rib_phase = (xx - float(x0)) / rib_period
        raised = 0.5 + 0.5 * np.cos(2.0 * np.pi * rib_phase)
        groove = 1.0 - raised
        fine = 0.5 + 0.5 * np.cos(2.0 * np.pi * rib_phase * 2.0)
        fiber = 0.5 + 0.5 * np.sin(2.0 * np.pi * (yy / 17.0 + xx / 59.0))
        shade = 0.91 + 0.09 * raised - 0.025 * groove + 0.015 * fine + 0.010 * fiber
    elif material == "linen":
        warp_period = max(4.0, panel_w / 22.0)
        weft_period = max(4.0, panel_h / 26.0)
        warp = 0.5 + 0.5 * np.sin(2.0 * np.pi * (xx - float(x0)) / warp_period)
        weft = 0.5 + 0.5 * np.sin(2.0 * np.pi * (yy - float(y0)) / weft_period)
        diagonal_fiber = 0.5 + 0.5 * np.sin(2.0 * np.pi * (xx / 37.0 + yy / 29.0))
        shade = 0.88 + 0.08 * warp + 0.08 * weft + 0.035 * diagonal_fiber
    elif material == "terry":
        loop_x = 0.5 + 0.5 * np.sin(2.0 * np.pi * (xx - float(x0)) / max(6.0, panel_w / 15.0))
        loop_y = 0.5 + 0.5 * np.cos(2.0 * np.pi * (yy - float(y0)) / max(6.0, panel_h / 16.0))
        loops = 0.55 * loop_x + 0.45 * loop_y
        speckle = 0.5 + 0.5 * np.sin(2.0 * np.pi * (xx / 11.0 + yy / 17.0))
        pile = 0.5 + 0.5 * np.cos(2.0 * np.pi * (xx / 19.0 - yy / 23.0))
        shade = 0.91 + 0.055 * loops + 0.035 * speckle + 0.025 * pile
    elif material == "quilted":
        cell = max(12.0, min(panel_w / 5.2, panel_h / 5.7) * max(0.25, scale))
        sx = cell
        sy = cell * 0.92
        row = np.floor((yy - float(y0)) / sy)
        row_offset = (np.mod(row, 2.0) * 0.5) * sx
        lx = np.mod(xx - float(x0) - row_offset + 0.5 * sx, sx) - 0.5 * sx
        ly = np.mod(yy - float(y0) + 0.5 * sy, sy) - 0.5 * sy
        radius = ((np.abs(lx) / (0.45 * sx)) ** 4 + (np.abs(ly) / (0.42 * sy)) ** 4) ** 0.25
        dome = np.exp(-(radius / 0.78) ** 2)
        valley = np.exp(-((radius - 1.0) / 0.26) ** 2)
        diagonal_light = 0.5 + 0.5 * np.clip((-0.55 * lx - ly) / max(1.0, cell), -1.0, 1.0)
        micro = 0.5 + 0.5 * np.sin(2.0 * np.pi * (xx / 31.0 + yy / 37.0))
        shade = 0.915 + 0.130 * dome + 0.035 * diagonal_light + 0.006 * micro - 0.075 * valley
    elif material == "waffle":
        cell = max(10.0, min(panel_w / 9.0, panel_h / 9.0) * max(0.25, scale))
        wx = np.mod(xx - float(x0), cell) / cell - 0.5
        wy = np.mod(yy - float(y0), cell) / cell - 0.5
        border = np.maximum(np.abs(wx), np.abs(wy))
        # Thin crisp seam lines along cell borders; soft raised square centers.
        seam = np.exp(-(((0.5 - border) / 0.07) ** 2))
        dome = np.exp(-((wx ** 2 + wy ** 2) / (2.0 * 0.20 ** 2)))
        sheen = np.clip(-0.6 * wx - wy, -1.0, 1.0)
        micro = 0.5 + 0.5 * np.sin(2.0 * np.pi * (xx / 29.0 + yy / 41.0))
        shade = 1.0 - 0.150 * seam + 0.045 * dome + 0.020 * dome * sheen + 0.006 * micro
    else:
        raise ValueError(f"Unsupported material panel: {material}")

    active = alpha[..., 0] > 0.02
    if np.any(active):
        shade_mean = float(np.average(shade[active], weights=np.maximum(alpha[..., 0][active], 1e-4)))
    else:
        shade_mean = float(shade.mean())
    clip_bounds = {
        "corduroy": (0.91, 1.09),
        "linen": (0.90, 1.10),
        "terry": (0.90, 1.10),
        "quilted": (0.86, 1.16),
        "waffle": (0.82, 1.10),
    }
    clip_min, clip_max = clip_bounds[material]
    active_luma01 = None
    if material in {"quilted", "waffle"} and np.any(active):
        luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        active_luma01 = float(np.average(luma[active], weights=np.maximum(alpha[..., 0][active], 1e-4))) / 255.0
    if material == "quilted" and active_luma01 is not None:
        dark_boost = 1.0 + 1.25 * np.clip((0.58 - active_luma01) / 0.35, 0.0, 1.0)
        shade = shade_mean + (shade - shade_mean) * dark_boost
        if active_luma01 < 0.58:
            clip_min, clip_max = 0.70, 1.34
    if contrast != 1.0:
        contrast = max(0.1, contrast)
        clip_min = max(0.60, 1.0 - (1.0 - clip_min) * contrast)
        clip_max = min(1.45, 1.0 + (clip_max - 1.0) * contrast)
    shade = shade / max(1e-4, shade_mean)
    if contrast != 1.0:
        shade = 1.0 + (shade - 1.0) * contrast
    shade = np.clip(shade, clip_min, clip_max)
    if active_luma01 is not None and active_luma01 > 0.55:
        # Bright fabric: cap highlights per pixel below saturation so shadow
        # grooves carry the texture instead of blown-out specular peaks.
        luma_px = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        shade = np.minimum(shade, 250.0 / np.maximum(luma_px, 1.0))
        if material == "waffle":
            # Attenuate texture amplitude in the host's own shadow regions so
            # grooves do not read as stains in already-dark fabric.
            atten = np.clip((luma_px / 255.0 - 0.30) / 0.35, 0.30, 1.0)
            shade = 1.0 + (shade - 1.0) * atten
    material_base = rgb
    if material == "quilted":
        blurred_base = np.asarray(
            Image.fromarray(np.clip(rgb, 0.0, 255.0).round().astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2.2)),
            dtype=np.float32,
        )
        material_base = rgb * 0.30 + blurred_base * 0.70
    textured = material_base * shade[..., None]

    # Add a subtle same-color pressed seam so the material panel reads as a local surface replacement.
    binary_mask = Image.fromarray((alpha[..., 0] > 0.05).astype(np.uint8) * 255, mode="L")
    edge = np.asarray(binary_mask.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=1.1)), dtype=np.float32)
    edge = (edge / 255.0)[..., None] * np.minimum(alpha, 1.0)
    textured = textured * (1.0 - 0.10 * edge)

    out = rgb * (1.0 - alpha) + textured * alpha
    return Image.fromarray(out.clip(0, 255).round().astype(np.uint8), mode="RGB")


def make_corduroy_reference(image: Image.Image, mask: Image.Image, opacity: float) -> Image.Image:
    return make_material_panel_reference(image, mask, opacity, "corduroy")


def soften_mask(
    mask: Image.Image,
    edge_feather_radius: float = 0.0,
    top_feather_frac: float = 0.0,
    top_feather_min_alpha: float = 0.0,
) -> Image.Image:
    if edge_feather_radius > 0.0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=edge_feather_radius))
    if top_feather_frac <= 0.0:
        return mask

    bbox = mask.getbbox()
    if bbox is None:
        return mask
    _, y0, _, y1 = bbox
    height = max(1, y1 - y0)
    feather_h = max(1, int(round(height * top_feather_frac)))
    arr = np.asarray(mask.convert("L"), dtype=np.float32)
    min_alpha = max(0.0, min(1.0, top_feather_min_alpha))
    for y in range(y0, min(y1, y0 + feather_h)):
        t = (y - y0) / float(feather_h)
        arr[y, :] *= min_alpha + (1.0 - min_alpha) * t
    return Image.fromarray(arr.clip(0, 255).round().astype(np.uint8), mode="L")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed decal mask and colored reference image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Reference image with decal composited.")
    parser.add_argument("--mask-output", required=True, type=Path)
    parser.add_argument("--overlay-output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--box", required=True, help="Normalized x0,y0,x1,y1 decal box.")
    parser.add_argument(
        "--shape",
        choices=(
            "rectangle",
            "slanted_rectangle",
            "corduroy_panel",
            "linen_panel",
            "terry_panel",
            "quilted_panel",
            "waffle_panel",
            "ellipse",
            "orange_fruit",
            "heart",
            "star",
            "printed_star",
            "leaf",
            "stripes",
            "dots",
            "cross",
        ),
        default="heart",
    )
    parser.add_argument("--slant-x", type=float, default=0.0, help="For slanted_rectangle, shift top edge right and bottom edge left as a fraction of box width.")
    parser.add_argument("--perspective-y", type=float, default=0.0, help="For slanted_rectangle, make the top narrower and bottom wider as a fraction of box width.")
    parser.add_argument("--edge-feather-radius", type=float, default=0.0)
    parser.add_argument("--top-feather-frac", type=float, default=0.0)
    parser.add_argument("--top-feather-min-alpha", type=float, default=0.0)
    parser.add_argument("--color", default="red")
    parser.add_argument("--opacity", type=float, default=0.92)
    parser.add_argument("--material-contrast", type=float, default=1.0, help="Amplitude multiplier for material panel shading (1.0 = legacy).")
    parser.add_argument("--material-scale", type=float, default=1.0, help="Spatial period multiplier for material panel texture (1.0 = legacy).")
    parser.add_argument("--host-mask", type=Path, default=None, help="Optional semantic host mask; material masks on dark hosts (where the bright-host luminance clip cannot apply) are intersected with it to prevent spill.")
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
    draw_shape(mask_draw, args.shape, pixel_box, fill=255, slant_x=args.slant_x, perspective_y=args.perspective_y)
    mask = soften_mask(
        mask,
        edge_feather_radius=args.edge_feather_radius,
        top_feather_frac=args.top_feather_frac,
        top_feather_min_alpha=args.top_feather_min_alpha,
    )

    if args.shape in {"corduroy_panel", "linen_panel", "terry_panel", "quilted_panel", "waffle_panel"}:
        mask, bright_clipped = clip_material_mask_to_local_host(image, mask)
        if args.host_mask is not None and args.host_mask.is_file():
            # Tight (eroded) intersection for all hosts: dilation lets the
            # latent-feathered final mask paint over occluders (e.g. a table
            # edge in front of the pillow), breaking depth order.
            mask = intersect_mask_with_host(mask, args.host_mask, image.size, tight=True)
        if not bright_clipped:
            # Dark hosts get a chroma-consistency guard against semantic-mask
            # over-segmentation spilling onto differently colored background.
            mask = clip_mask_to_host_color(image, mask)
        reference = make_material_panel_reference(
            image,
            mask,
            args.opacity,
            args.shape.removesuffix("_panel"),
            contrast=args.material_contrast,
            scale=args.material_scale,
        )
    elif args.shape == "orange_fruit":
        fruit_mask = mask
        shadow_mask = make_orange_shadow_mask(image.size, pixel_box)
        reference = make_orange_fruit_reference(image, fruit_mask, shadow_mask, pixel_box, args.opacity)
        mask = Image.fromarray(
            np.maximum(np.asarray(fruit_mask), np.asarray(shadow_mask)).astype(np.uint8),
            "L",
        )
    elif args.shape == "printed_star":
        reference = make_printed_star_reference(image, mask, color, args.opacity)
    else:
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
        "slant_x": float(args.slant_x),
        "perspective_y": float(args.perspective_y),
        "edge_feather_radius": float(args.edge_feather_radius),
        "top_feather_frac": float(args.top_feather_frac),
        "top_feather_min_alpha": float(args.top_feather_min_alpha),
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
