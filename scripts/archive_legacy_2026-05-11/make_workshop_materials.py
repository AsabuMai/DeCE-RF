from __future__ import annotations

import argparse
import io
import textwrap
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REGULAR)
    return ImageFont.truetype(str(path), size=size)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    box_w: int,
    font_obj: ImageFont.ImageFont,
    fill: str,
    line_gap: int = 8,
    max_lines: int | None = None,
) -> int:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        probe = word if not line else f"{line} {word}"
        if draw.textbbox((0, 0), probe, font=font_obj)[2] <= box_w:
            line = probe
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    if max_lines is not None:
        lines = lines[:max_lines]
    x, y = xy
    line_h = draw.textbbox((0, 0), "Ag", font=font_obj)[3] + line_gap
    for line in lines:
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += line_h
    return y


def rounded_rect(draw: ImageDraw.ImageDraw, xy, radius: int, fill: str, outline: str | None = None, width: int = 2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def latex_image(expr: str, fontsize: int, color: str = "#0f172a") -> Image.Image:
    fig = plt.figure(figsize=(0.01, 0.01), dpi=220)
    fig.text(0, 0, f"${expr}$", fontsize=fontsize, color=color)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def paste_rgba(canvas: Image.Image, image: Image.Image, xy: tuple[int, int]) -> None:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    canvas.paste(image, xy, image)


def fit_rgba_width(image: Image.Image, max_w: int) -> Image.Image:
    if image.width <= max_w:
        return image
    scale = max_w / image.width
    return image.resize((max_w, max(1, int(image.height * scale))), Image.Resampling.LANCZOS)


def draw_latex_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    w: int,
    title: str,
    equations: list[str],
    eq_fontsize: int = 34,
    min_h: int = 260,
) -> int:
    x, y = xy
    rendered = [fit_rgba_width(latex_image(eq, eq_fontsize), w - 72) for eq in equations]
    h = max(min_h, 88 + sum(img.height + 22 for img in rendered))
    rounded_rect(draw, (x, y, x + w, y + h), 18, "#ffffff", "#cbd5e1", 3)
    draw.text((x + 28, y + 22), title, font=font(24, bold=True), fill="#0f172a")
    yy = y + 82
    for img in rendered:
        paste_rgba(canvas, img, (x + 30, yy))
        yy += img.height + 22
    return y + h + 26


def paste_cover(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0
    img = image.convert("RGB")
    scale = max(box_w / img.width, box_h / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - box_w) // 2
    top = (resized.height - box_h) // 2
    cropped = resized.crop((left, top, left + box_w, top + box_h))
    canvas.paste(cropped, (x0, y0))


def paste_contain(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int], fill: str = "#ffffff") -> None:
    x0, y0, x1, y1 = box
    box_w, box_h = x1 - x0, y1 - y0
    bg = Image.new("RGB", (box_w, box_h), fill)
    img = image.convert("RGB")
    scale = min(box_w / img.width, box_h / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    bg.paste(resized, ((box_w - resized.width) // 2, (box_h - resized.height) // 2))
    canvas.paste(bg, (x0, y0))


def grid_tile(image: Image.Image, index: int, ncols: int, header_h: int = 0) -> Image.Image:
    """Crop one column from a fixed-width comparison grid."""
    x0 = round(index * image.width / ncols)
    x1 = round((index + 1) * image.width / ncols)
    return image.crop((x0, header_h, x1, image.height))


def draw_formula_card(draw: ImageDraw.ImageDraw, xy: tuple[int, int], w: int, title: str, lines: list[str]) -> int:
    x, y = xy
    rounded_rect(draw, (x, y, x + w, y + 250), 18, "#f7fafc", "#cbd5e1", 3)
    draw.text((x + 28, y + 22), title, font=font(30, bold=True), fill="#0f172a")
    yy = y + 78
    for line in lines:
        draw.text((x + 32, yy), line, font=font(27, mono=True), fill="#1e293b")
        yy += 44
    return y + 280


def draw_pipeline(draw: ImageDraw.ImageDraw, x: int, y: int, w: int) -> int:
    boxes = [
        ("Source\nimage x0", "#e0f2fe"),
        ("Source\ninversion", "#dcfce7"),
        ("RF h-edit\nODE", "#fef3c7"),
        ("Edited\nimage", "#fae8ff"),
    ]
    bw = (w - 90) // 4
    for i, (label, fill) in enumerate(boxes):
        bx = x + i * (bw + 30)
        rounded_rect(draw, (bx, y, bx + bw, y + 145), 16, fill, "#334155", 3)
        lines = label.split("\n")
        for j, line in enumerate(lines):
            tw = draw.textbbox((0, 0), line, font=font(28, bold=True))[2]
            draw.text((bx + (bw - tw) // 2, y + 38 + j * 38), line, font=font(28, bold=True), fill="#0f172a")
        if i < len(boxes) - 1:
            ax0 = bx + bw + 6
            ay = y + 72
            ax1 = bx + bw + 24
            draw.line((ax0, ay, ax1, ay), fill="#334155", width=6)
            draw.polygon([(ax1, ay - 12), (ax1, ay + 12), (ax1 + 18, ay)], fill="#334155")
    return y + 175


def make_poster(result_image: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = 2550, 3600
    canvas = Image.new("RGB", (W, H), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    result = Image.open(result_image)
    cross_method_path = ROOT / "outputs/sunglasses_cross_method_grid.png"
    quality_path = ROOT / "outputs/archive/2026-04-24-exploration/sunglasses_final_quality_candidates.png"
    mask_grid_path = ROOT / "outputs/sunglasses_cross_method_mask_grid.png"
    cross_method = Image.open(cross_method_path) if cross_method_path.exists() else None
    quality_grid = Image.open(quality_path) if quality_path.exists() else None
    mask_grid = Image.open(mask_grid_path) if mask_grid_path.exists() else None

    navy = "#0f172a"
    blue = "#2563eb"
    teal = "#0f766e"
    orange = "#ea580c"
    red = "#dc2626"
    slate = "#334155"
    muted = "#64748b"

    def title_text(x: int, y: int, text: str, color: str = navy) -> int:
        draw.text((x, y), text, font=font(34, bold=True), fill=color)
        return y + 54

    def panel(x: int, y: int, w: int, h: int, title: str, accent: str) -> int:
        rounded_rect(draw, (x, y, x + w, y + h), 16, "#ffffff", "#d9e2ef", 2)
        draw.rectangle((x, y, x + 10, y + h), fill=accent)
        draw.text((x + 32, y + 24), title, font=font(28, bold=True), fill=navy)
        return y + 82

    def bullet_list(x: int, y: int, w: int, items: list[str], color: str, size: int = 20, gap_y: int = 17) -> int:
        for item in items:
            draw.ellipse((x, y + 9, x + 14, y + 23), fill=color)
            y = draw_wrapped(draw, (x + 30, y), item, w - 30, font(size), slate, 5, max_lines=3) + gap_y
        return y

    def mini_label(x: int, y: int, label: str, body: str, w: int, accent: str) -> int:
        rounded_rect(draw, (x, y, x + w, y + 92), 10, "#f8fafc", "#e2e8f0", 2)
        draw.rectangle((x, y, x + 7, y + 92), fill=accent)
        draw.text((x + 24, y + 18), label, font=font(19, bold=True), fill=navy)
        draw_wrapped(draw, (x + 210, y + 17), body, w - 235, font(18), slate, 4, max_lines=2)
        return y + 108

    def draw_image_cell(
        image: Image.Image,
        box: tuple[int, int, int, int],
        label: str,
        note: str,
        accent: str,
        contain_fill: str = "#ffffff",
    ) -> None:
        x0, y0, x1, y1 = box
        rounded_rect(draw, (x0, y0, x1, y1), 14, "#ffffff", "#d9e2ef", 2)
        img_h = y1 - y0 - 72
        paste_contain(canvas, image, (x0 + 14, y0 + 14, x1 - 14, y0 + img_h), contain_fill)
        draw.rectangle((x0 + 14, y0 + img_h + 17, x0 + 22, y1 - 18), fill=accent)
        draw.text((x0 + 34, y0 + img_h + 15), label, font=font(21, bold=True), fill=navy)
        draw.text((x0 + 34, y0 + img_h + 48), note, font=font(16), fill=slate)

    def draw_comparison_strip(x: int, y: int, w: int) -> int:
        rounded_rect(draw, (x, y, x + w, y + 980), 18, "#ffffff", "#d9e2ef", 2)
        draw.text((x + 34, y + 24), "Result comparison under mask/guidance choices", font=font(29, bold=True), fill=navy)
        draw_wrapped(
            draw,
            (x + 34, y + 66),
            "The four panels show the reference, the selected RF h-Edit result, and two typical failure modes.",
            w - 68,
            font(18),
            slate,
            4,
            max_lines=2,
        )
        cell_gap = 22
        cell_w = (w - 68 - cell_gap) // 2
        cell_h = 405
        y0 = y + 136
        if quality_grid is not None:
            source = grid_tile(quality_grid, 0, 6, 48)
            ours = result
            alternate = grid_tile(quality_grid, 2, 6, 48)
            over_edit = grid_tile(quality_grid, 5, 6, 48)
        elif cross_method is not None:
            source = grid_tile(cross_method, 0, 7, 48)
            ours = result
            alternate = grid_tile(cross_method, 3, 7, 48)
            over_edit = grid_tile(cross_method, 4, 7, 48)
        else:
            source = result
            ours = result
            alternate = result
            over_edit = result
        draw_image_cell(source, (x + 34, y0, x + 34 + cell_w, y0 + cell_h), "Source image", "preserve identity and background", blue)
        draw_image_cell(ours, (x + 34 + cell_w + cell_gap, y0, x + 34 + 2 * cell_w + cell_gap, y0 + cell_h), "Best RF h-Edit", "selected local edit result", teal)
        y1 = y0 + cell_h + 28
        draw_image_cell(alternate, (x + 34, y1, x + 34 + cell_w, y1 + cell_h), "Placement drift", "glasses move under mask choice", orange)
        draw_image_cell(over_edit, (x + 34 + cell_w + cell_gap, y1, x + 34 + 2 * cell_w + cell_gap, y1 + cell_h), "Identity drift", "strong edit changes the face", red)
        return y + 1015

    def draw_quality_strip(x: int, y: int, w: int) -> int:
        rounded_rect(draw, (x, y, x + w, y + 440), 18, "#ffffff", "#d9e2ef", 2)
        draw.text((x + 34, y + 24), "Quality trade-off sweep", font=font(29, bold=True), fill=navy)
        draw_wrapped(
            draw,
            (x + 34, y + 65),
            "Different support choices move the edit location and alter facial structure; this is the bottleneck the next step targets.",
            w - 68,
            font(18),
            slate,
            4,
            max_lines=2,
        )
        labels = ["source", "cleaner", "shifted", "over-edited"]
        notes = ["reference", "best balance", "placement drift", "identity drift"]
        colors = [blue, teal, orange, red]
        if quality_grid is not None:
            imgs = [
                grid_tile(quality_grid, 0, 6, 48),
                grid_tile(quality_grid, 1, 6, 48),
                grid_tile(quality_grid, 2, 6, 48),
                grid_tile(quality_grid, 5, 6, 48),
            ]
        else:
            imgs = [result, result, result, result]
        cell_gap = 18
        cell_w = (w - 68 - 3 * cell_gap) // 4
        y0 = y + 124
        for i, img in enumerate(imgs):
            x0 = x + 34 + i * (cell_w + cell_gap)
            draw_image_cell(img, (x0, y0, x0 + cell_w, y0 + 270), labels[i], notes[i], colors[i])
        return y + 475

    def arrow(x0: int, y0: int, x1: int, y1: int, color: str, width: int = 5) -> None:
        draw.line((x0, y0, x1, y1), fill=color, width=width)
        dx = 1 if x1 >= x0 else -1
        draw.polygon([(x1, y1), (x1 - 18 * dx, y1 - 10), (x1 - 18 * dx, y1 + 10)], fill=color)

    def draw_rf_path_diagram(x: int, y: int, w: int) -> int:
        rounded_rect(draw, (x, y, x + w, y + 128), 12, "#f8fafc", "#e2e8f0", 2)
        draw.text((x + 22, y + 16), "RF path", font=font(19, bold=True), fill=navy)
        x0, x1, yy = x + 165, x + w - 95, y + 72
        draw.line((x0, yy, x1, yy), fill="#94a3b8", width=6)
        for px, label, color in [
            (x0, "source x0", blue),
            ((x0 + x1) // 2, "current xt", teal),
            (x1, "noise x1", orange),
        ]:
            draw.ellipse((px - 15, yy - 15, px + 15, yy + 15), fill=color)
            tw = draw.textbbox((0, 0), label, font=font(16))[2]
            draw.text((px - tw // 2, yy + 25), label, font=font(16), fill=slate)
        return y + 150

    def draw_velocity_split_diagram(x: int, y: int, w: int) -> int:
        rounded_rect(draw, (x, y, x + w, y + 168), 12, "#f8fafc", "#e2e8f0", 2)
        draw.text((x + 22, y + 16), "Velocity split", font=font(19, bold=True), fill=navy)
        boxes = [
            ("source\nbase", blue),
            ("reconstruct", teal),
            ("edit", orange),
        ]
        bx, by = x + 32, y + 62
        bw, bh = 150, 72
        join_x = x + w - 220
        for i, (label, color) in enumerate(boxes):
            px = bx + i * 180
            rounded_rect(draw, (px, by, px + bw, by + bh), 10, "#ffffff", "#cbd5e1", 2)
            draw.rectangle((px, by, px + 8, by + bh), fill=color)
            lines = label.split("\n")
            for j, line in enumerate(lines):
                draw.text((px + 22, by + 14 + j * 24), line, font=font(17, bold=True), fill=navy)
            arrow(px + bw + 8, by + bh // 2, join_x, by + bh // 2, "#94a3b8", 4)
        draw.ellipse((join_x - 20, by + bh // 2 - 20, join_x + 20, by + bh // 2 + 20), fill=navy)
        draw.text((join_x - 7, by + bh // 2 - 17), "+", font=font(24, bold=True), fill="#ffffff")
        arrow(join_x + 28, by + bh // 2, x + w - 42, by + bh // 2, navy, 5)
        draw.text((x + w - 118, by + 10), "update", font=font(18, bold=True), fill=navy)
        return y + 190

    def draw_mask_gate_diagram(x: int, y: int, w: int) -> int:
        rounded_rect(draw, (x, y, x + w, y + 150), 12, "#f8fafc", "#e2e8f0", 2)
        draw.text((x + 22, y + 16), "Mask gate", font=font(19, bold=True), fill=navy)
        gx, gy, cell = x + 40, y + 58, 24
        for r in range(4):
            for c in range(7):
                fill = "#dbeafe"
                if 2 <= c <= 4 and 1 <= r <= 2:
                    fill = "#fed7aa"
                draw.rectangle((gx + c * cell, gy + r * cell, gx + (c + 1) * cell - 3, gy + (r + 1) * cell - 3), fill=fill, outline="#cbd5e1")
        draw.text((gx + 200, gy + 2), "edit support M", font=font(18, bold=True), fill=navy)
        draw_wrapped(
            draw,
            (gx + 200, gy + 34),
            "orange region receives edit correction; blue region is preserved.",
            w - 250,
            font(17),
            slate,
            4,
            max_lines=2,
        )
        return y + 172

    def draw_math_core_academic(x: int, y: int, w: int) -> int:
        rows = [
            (
                "1. Read the current state in clean-image space",
                r"x_t=(1-t)x_0+t x_1,\qquad \hat{x}_0(x_t,t)=x_t-t\,v_\theta(x_t,t)",
                "The clean estimate is where source preservation and target-edit objectives are measured.",
                blue,
                15,
            ),
            (
                "2. Separate the reverse ODE into base, reconstruction, and edit",
                r"\dot{x}_t=v_{\rm src}(x_t,t)+u_{\rm rec}(x_t,t)+u_{\rm edit}(x_t,t)",
                "Source faithfulness and target transformation are controlled by different fields.",
                teal,
                16,
            ),
            (
                "3. Build correction fields from explicit objectives",
                r"E_{\rm rec}=\frac{1}{2}\|(1-M)\odot(\hat{x}_0-x^{\rm src})\|^2,\quad u_{\rm edit}\sim\beta(t)\nabla_{x_t}E_{\rm edit}",
                "The reconstruction term protects non-edit regions; the editing energy defines target control.",
                orange,
                11,
            ),
        ]
        yy = y
        row_h = 138
        for title, expr, note, accent, eq_size in rows:
            rounded_rect(draw, (x, yy, x + w, yy + row_h), 12, "#f8fafc", "#e2e8f0", 2)
            draw.rectangle((x, yy, x + 7, yy + row_h), fill=accent)
            draw.text((x + 24, yy + 17), title, font=font(18, bold=True), fill=navy)
            eq = fit_rgba_width(latex_image(expr, eq_size, navy), w - 70)
            paste_rgba(canvas, eq, (x + 24, yy + 45))
            draw_wrapped(draw, (x + 24, yy + 112), note, w - 48, font(13), slate, 3, max_lines=2)
            yy += row_h + 12

        rounded_rect(draw, (x, yy, x + w, yy + 150), 12, "#ffffff", "#cbd5e1", 2)
        draw.text((x + 24, yy + 18), "Term roles", font=font(18, bold=True), fill=navy)
        chips = [
            ("v_src", "source-conditioned RF base velocity", blue),
            ("u_rec", "faithfulness / non-edit preservation correction", teal),
            ("u_edit", "target-seeking editing correction", orange),
        ]
        cy = yy + 58
        for label, body, color in chips:
            draw.ellipse((x + 24, cy + 5, x + 42, cy + 23), fill=color)
            draw.text((x + 56, cy), label, font=font(15, bold=True, mono=True), fill=navy)
            draw.text((x + 180, cy + 1), body, font=font(15), fill=slate)
            cy += 30
        return yy + 172

    # Header.
    draw.rectangle((0, 0, W, 390), fill=navy)
    draw.text((100, 52), "RF h-Edit", font=font(88, bold=True), fill="#ffffff")
    draw.text((102, 158), "Reconstruction-aware Rectified Flow image editing", font=font(43, bold=True), fill="#c7d2fe")
    draw.text((104, 248), "Wu et al.  |  SD3 Rectified Flow prototype  |  Workshop poster", font=font(26), fill="#e2e8f0")
    draw.text((104, 304), "Core idea: use source-aware dynamics for faithfulness, then add a separate target-edit field.", font=font(27), fill="#bfdbfe")

    margin = 100
    top = 470
    footer_y = H - 118
    gap = 70
    left_w = (W - 2 * margin - gap) // 2
    right_w = left_w
    left_x = margin
    right_x = left_x + left_w + gap

    # Left column: motivation and method.
    x, y = left_x, top
    y = title_text(x, y, "Problem and idea")
    body_y = panel(x, y, left_w, 360, "Why decouple?", blue)
    yy = bullet_list(
        x + 34,
        body_y,
        left_w - 68,
        [
            "Target guidance can add the concept, but often drifts identity and geometry.",
            "Local masks protect background, yet overly narrow masks can suppress the edit.",
            "RF h-Edit makes these two roles explicit in the ODE.",
        ],
        blue,
        21,
        26,
    )
    y += 420

    body_y = panel(x, y, left_w, 735, "Method in one panel", teal)
    draw_math_core_academic(x + 34, body_y + 4, left_w - 68)
    y += 785

    body_y = panel(x, y, left_w, 590, "Prototype pipeline", orange)
    steps = [
        ("1", "Invert source", "save the RF source trajectory"),
        ("2", "Estimate support", "attention / external local mask"),
        ("3", "Integrate ODE", "source base + rec + edit corrections"),
        ("4", "Preserve", "blend non-edit regions to the source trajectory"),
    ]
    yy = body_y + 12
    for num, name, desc in steps:
        rounded_rect(draw, (x + 34, yy, x + left_w - 34, yy + 92), 14, "#f8fafc", "#e2e8f0", 2)
        draw.ellipse((x + 58, yy + 22, x + 104, yy + 68), fill=orange)
        tw = draw.textbbox((0, 0), num, font=font(22, bold=True))[2]
        draw.text((x + 58 + (46 - tw) // 2, yy + 31), num, font=font(22, bold=True), fill="#ffffff")
        draw.text((x + 126, yy + 18), name, font=font(22, bold=True), fill=navy)
        draw.text((x + 126, yy + 54), desc, font=font(18), fill=slate)
        yy += 110
    draw_wrapped(draw, (x + 34, yy + 12), "Current run: local support + source trajectory preservation.", left_w - 68, font(19), slate, 5, max_lines=2)
    y += 635

    y = title_text(x, y, "Limitation and next step")
    body_y = panel(x, y, left_w, 930, "What remains hard?", red)
    yy = body_y + 18
    yy = bullet_list(
        x + 34,
        yy,
        left_w - 68,
        [
            "Mask tuning alone creates a placement-versus-faithfulness trade-off.",
            "The next bottleneck is source feature reuse during target velocity evaluation.",
        ],
        red,
        20,
        26,
    )
    draw.text((x + 34, yy + 72), "Next control lever", font=font(23, bold=True), fill=navy)
    yy += 120
    for label, body in [
        ("Source pass", "store source visual features"),
        ("Target pass", "reuse V / attention while evaluating target velocity"),
    ]:
        yy = mini_label(x + 34, yy, label, body, left_w - 68, teal) + 22
    draw.text((x + 34, yy + 34), "Minimal validation", font=font(23, bold=True), fill=navy)
    yy += 82
    for label, body in [
        ("Keep fixed", "same seed, mask, source inversion, and target prompt"),
        ("Measure", "background drift, edit placement, and facial identity"),
    ]:
        yy = mini_label(x + 34, yy, label, body, left_w - 68, red) + 22

    # Right column: visual evidence first, then the concise claim.
    x, y = right_x, top
    y = title_text(x, y, "Main visual result")
    y = draw_comparison_strip(x, y, right_w)

    y = title_text(x, y, "Evidence")
    body_y = panel(x, y, right_w, 690, "What the ablations say", blue)
    yy = body_y + 18
    for label, body in [
        ("Direct target", "adds the object, but redraws identity and geometry"),
        ("Mask only", "protects the scene, but can miss or fragment the glasses"),
        ("Trajectory", "keeps background close while limiting non-edit drift"),
        ("Best recipe", "source trajectory + local support + explicit reconstruction term"),
    ]:
        yy = mini_label(x + 34, yy, label, body, right_w - 68, blue)
        yy += 10
    y += 745

    y = title_text(x, y, "Workshop takeaway")
    body_y = panel(x, y, right_w, 910, "Claim and next lever", teal)
    draw_wrapped(
        draw,
        (x + 42, body_y + 8),
        "RF h-Edit is a useful scaffold because it separates source reconstruction from target editing. The remaining bottleneck is not the ODE split itself; it is source feature reuse while evaluating the target velocity.",
        right_w - 84,
        font(24),
        slate,
        9,
        max_lines=6,
    )
    eq_box_y = body_y + 292
    rounded_rect(draw, (x + 42, eq_box_y, x + right_w - 42, eq_box_y + 126), 12, "#f8fafc", "#e2e8f0", 2)
    eq = fit_rgba_width(latex_image(r"\dot{x}_t=v_{\rm src}(x_t,t)+u_{\rm rec}(x_t,t)+u_{\rm edit}(x_t,t)", 19, navy), right_w - 160)
    paste_rgba(canvas, eq, (x + 78, eq_box_y + 28))
    draw.text((x + 78, eq_box_y + 91), "The poster's main message is the separation of these three fields.", font=font(16), fill=slate)

    yy = body_y + 500
    for label, body, color in [
        ("Math", "explicit rec/edit terms", blue),
        ("Mask", "local support is necessary but insufficient", orange),
        ("Next", "attention/feature reuse during target pass", teal),
    ]:
        rounded_rect(draw, (x + 42, yy, x + right_w - 42, yy + 58), 10, "#f8fafc", "#e2e8f0", 2)
        draw.ellipse((x + 62, yy + 18, x + 82, yy + 38), fill=color)
        draw.text((x + 102, yy + 14), label, font=font(19, bold=True), fill=navy)
        draw.text((x + 225, yy + 15), body, font=font(18), fill=slate)
        yy += 72

    draw.line((margin, footer_y, W - margin, footer_y), fill="#cbd5e1", width=3)
    draw.text((margin, footer_y + 30), "Generated from current RF h-Edit project outputs. Main RF h-Edit panel uses the original best panda-sunglasses run.", font=font(23), fill=muted)

    png_path = out_dir / "rf_h_edit_workshop_poster_vertical.png"
    pdf_path = out_dir / "rf_h_edit_workshop_poster_vertical.pdf"
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=150.0)
    return png_path


def make_slide_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (1920, 1080), "#f8fafc")
    return canvas, ImageDraw.Draw(canvas)


def slide_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None = None) -> None:
    draw.rectangle((0, 0, 1920, 150), fill="#0f172a")
    draw.text((72, 38), title, font=font(54, bold=True), fill="#ffffff")
    if subtitle:
        draw.text((74, 102), subtitle, font=font(26), fill="#c7d2fe")


def make_slides(result_image: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = Image.open(result_image)
    slide_paths: list[Path] = []

    c, d = make_slide_canvas()
    slide_header(d, "RF h-Edit", "Reconstruction-aware Rectified Flow image editing")
    d.text((90, 225), "One-sentence idea", font=font(42, bold=True), fill="#0f172a")
    draw_wrapped(
        d,
        (90, 292),
        "Use the source RF field as a base, then add separate reconstruction and editing corrections.",
        770,
        font(38),
        "#334155",
        12,
    )
    img = latex_image(r"\dot{x}_t=v_{\rm src}+u_{\rm rec}+u_{\rm edit}", 28, "#0f172a")
    rounded_rect(d, (90, 520, 875, 665), 20, "#ffffff", "#cbd5e1", 3)
    paste_rgba(c, img, (130, 555))
    d.text((90, 760), "Demo task", font=font(38, bold=True), fill="#0f172a")
    draw_wrapped(d, (90, 820), "Local insertion: add sunglasses while preserving the panda and forest.", 770, font(34), "#334155", 10)
    rounded_rect(d, (1025, 205, 1810, 1015), 24, "#ffffff", "#cbd5e1", 3)
    paste_contain(c, result, (1060, 240, 1775, 910), "#ffffff")
    d.text((1060, 940), "Best current SD3 result", font=font(29, bold=True), fill="#0f172a")
    d.text((1060, 980), "mask + source trajectory preservation", font=font(25), fill="#475569")
    p = out_dir / "rf_h_edit_2min_slide_01.png"
    c.save(p)
    slide_paths.append(p)

    c, d = make_slide_canvas()
    slide_header(d, "Mathematical idea", "Do not collapse reconstruction and editing into one guidance term")
    y = draw_latex_card(
        c,
        d,
        (90, 225),
        820,
        "Rectified Flow path",
        [r"x_t=(1-t)x_0+t x_1", r"\hat{x}_0=x_t-t\,v_\theta(x_t,t)"],
        eq_fontsize=24,
        min_h=270,
    )
    draw_latex_card(
        c,
        d,
        (90, y + 20),
        820,
        "RF h-edit ODE",
        [r"\dot{x}_t=v_{\rm src}+u_{\rm rec}+u_{\rm edit}", r"u=-\Delta x_0/\max(t,\epsilon)"],
        eq_fontsize=24,
        min_h=285,
    )
    d.text((1030, 245), "Interpretation", font=font(42, bold=True), fill="#0f172a")
    notes = [
        ("v_src", "source-conditioned RF base field"),
        ("u_rec", "faithfulness correction"),
        ("u_edit", "target edit correction"),
    ]
    yy = 330
    for label, body in notes:
        rounded_rect(d, (1030, yy, 1785, yy + 125), 18, "#ffffff", "#cbd5e1", 2)
        d.text((1060, yy + 28), label, font=font(34, bold=True, mono=True), fill="#2563eb")
        d.text((1205, yy + 33), body, font=font(28), fill="#334155")
        yy += 160
    p = out_dir / "rf_h_edit_2min_slide_02.png"
    c.save(p)
    slide_paths.append(p)

    c, d = make_slide_canvas()
    slide_header(d, "Prototype pipeline", "Current SD3 implementation")
    draw_pipeline(d, 115, 230, 1690)
    d.text((120, 475), "Operational split", font=font(42, bold=True), fill="#0f172a")
    items = [
        ("1", "Invert", "follow source RF field to obtain a source trajectory"),
        ("2", "Mask", "derive local edit support from attention or proposal masks"),
        ("3", "Integrate", "run reverse ODE with source base, rec term, and edit term"),
        ("4", "Preserve", "blend non-edit regions toward the saved source trajectory"),
    ]
    y = 560
    for num, title, body in items:
        rounded_rect(d, (120, y, 1800, y + 95), 18, "#ffffff", "#cbd5e1", 2)
        d.ellipse((150, y + 25, 198, y + 73), fill="#2563eb")
        d.text((165, y + 31), num, font=font(25, bold=True), fill="#ffffff")
        d.text((230, y + 27), title, font=font(31, bold=True), fill="#0f172a")
        d.text((420, y + 30), body, font=font(29), fill="#334155")
        y += 115
    p = out_dir / "rf_h_edit_2min_slide_03.png"
    c.save(p)
    slide_paths.append(p)

    c, d = make_slide_canvas()
    slide_header(d, "Result and next step", "What the current evidence says")
    rounded_rect(d, (90, 220, 820, 950), 24, "#ffffff", "#cbd5e1", 3)
    paste_contain(c, result, (125, 255, 785, 855), "#ffffff")
    d.text((125, 875), "Local sunglasses insertion", font=font(28), fill="#475569")
    d.text((940, 225), "Takeaways", font=font(42, bold=True), fill="#0f172a")
    takeaways = [
        "The RF h-edit decomposition is working as a controllable framework.",
        "Masks and trajectory preservation give strong background protection.",
        "Mask tuning alone hits a quality/placement trade-off on local edits.",
        "Next: SD3 source-reference V/attention injection, inspired by RF-Solver, FireFlow, and ReFlex.",
    ]
    y = 310
    for b in takeaways:
        d.ellipse((950, y + 15, 970, y + 35), fill="#7c3aed")
        y = draw_wrapped(d, (990, y), b, 780, font(32), "#334155", 10) + 26
    img = latex_image(r"\dot{x}_t=v_{\rm src}+u_{\rm rec}+u_{\rm edit}", 22, "#0f172a")
    rounded_rect(d, (940, 875, 1790, 990), 18, "#ffffff", "#cbd5e1", 2)
    paste_rgba(c, img, (980, 908))
    p = out_dir / "rf_h_edit_2min_slide_04.png"
    c.save(p)
    slide_paths.append(p)
    return slide_paths


def slide_xml(image_rid: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      <p:pic>
        <p:nvPicPr><p:cNvPr id="2" name="slide image"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
        <p:blipFill><a:blip r:embed="{image_rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
        <p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def make_pptx(slides: list[Path], pptx_path: Path) -> None:
    pptx_path.parent.mkdir(parents=True, exist_ok=True)
    content_types = ['''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
''']
    for i in range(1, len(slides) + 1):
        content_types.append(f'  <Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>\n')
    content_types.append("</Types>")

    slide_ids = "\n".join(
        f'    <p:sldId id="{256 + i}" r:id="rId{i}"/>' for i in range(1, len(slides) + 1)
    )
    pres_rels = "\n".join(
        f'  <Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, len(slides) + 1)
    )

    with zipfile.ZipFile(pptx_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(content_types))
        z.writestr("_rels/.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>''')
        z.writestr("docProps/core.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{escape(pptx_path.stem)}</dc:title></cp:coreProperties>''')
        z.writestr("docProps/app.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex</Application></Properties>''')
        z.writestr("ppt/presentation.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldIdLst>
{slide_ids}
  </p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>''')
        z.writestr("ppt/_rels/presentation.xml.rels", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{pres_rels}
</Relationships>''')
        for i, slide_path in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml("rId1"))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/slide{i}.png"/>
</Relationships>''')
            z.write(slide_path, f"ppt/media/slide{i}.png")


def make_speaker_notes(out_dir: Path) -> Path:
    notes = """# RF h-Edit 2-minute intro speaker notes

## Slide 1
We are building RF h-Edit, a reconstruction-aware Rectified Flow image editing prototype. The goal is to make prompt edits while keeping the source image faithful. The current demo is a panda wearing sunglasses.

## Slide 2
The central idea is to keep the source RF velocity as the base field, then add two separate corrections. The reconstruction term protects source faithfulness, and the edit term pushes toward the target prompt. So the implementation uses x_dot_t = v_src + u_rec + u_edit.

## Slide 3
The pipeline first inverts the source image into the RF trajectory, then extracts a local edit support mask. During reverse ODE sampling, we use the source base field, add reconstruction and edit corrections, and preserve regions by blending back toward the saved source trajectory.

## Slide 4
The current best result shows that local masks plus trajectory preservation can protect the background and add sunglasses. The main limitation is that mask tuning alone creates a placement-quality trade-off. The next step is source-reference feature or attention injection in SD3, inspired by RF-Solver, FireFlow, and ReFlex.
"""
    path = out_dir / "rf_h_edit_2min_intro_speaker_notes.md"
    path.write_text(notes, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-image",
        type=Path,
        default=ROOT / "outputs/sunglasses_external_mask_traj_sd3/result.png",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "workshop_materials")
    args = parser.parse_args()

    out_dir = args.out_dir
    poster = make_poster(args.result_image, out_dir)
    print(f"poster: {poster}")
    print(f"poster_pdf: {poster.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
