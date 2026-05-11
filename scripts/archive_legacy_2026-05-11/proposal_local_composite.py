from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image).astype(np.uint8)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(path)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def parse_roi(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x0,y0,x1,y1 in normalized coordinates")
    return tuple(parts)  # type: ignore[return-value]


def roi_mask(shape: tuple[int, int], roi: tuple[float, float, float, float] | None) -> np.ndarray:
    h, w = shape
    mask = np.ones((h, w), dtype=np.uint8)
    if roi is None:
        return mask
    x0, y0, x1, y1 = roi
    out = np.zeros((h, w), dtype=np.uint8)
    ix0 = int(np.clip(round(x0 * w), 0, w))
    ix1 = int(np.clip(round(x1 * w), 0, w))
    iy0 = int(np.clip(round(y0 * h), 0, h))
    iy1 = int(np.clip(round(y1 * h), 0, h))
    out[iy0:iy1, ix0:ix1] = 1
    return out


def otsu_threshold(values: np.ndarray, floor: float) -> float:
    values_u8 = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(values_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(float(threshold) / 255.0, floor)


def keep_components(
    mask: np.ndarray,
    score: np.ndarray,
    keep: int,
    min_area: int,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps: list[tuple[float, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label
        mean_score = float(score[component].mean())
        # Prefer compact high-confidence changes over global tone shifts.
        comps.append((mean_score * np.sqrt(area), label))
    comps.sort(reverse=True)
    out = np.zeros_like(mask, dtype=np.uint8)
    for _, label in comps[:keep]:
        out[labels == label] = 1
    return out


def build_mask(
    source: np.ndarray,
    proposal: np.ndarray,
    roi: tuple[float, float, float, float] | None,
    threshold: float,
    keep: int,
    min_area: int,
    dilate: int,
    erode: int,
    blur: int,
    dark_bias: float,
) -> tuple[np.ndarray, dict[str, float]]:
    src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    prop_lab = cv2.cvtColor(proposal, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_diff = normalize01(np.linalg.norm(prop_lab - src_lab, axis=-1))

    src_luma = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    prop_luma = cv2.cvtColor(proposal, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    darker = np.clip(src_luma - prop_luma, 0.0, 1.0)
    score = normalize01(lab_diff + dark_bias * darker)

    region = roi_mask(score.shape, roi)
    masked_score = score * region
    auto_threshold = otsu_threshold(masked_score[region > 0], floor=threshold)
    binary = ((masked_score >= auto_threshold) & (region > 0)).astype(np.uint8)

    if erode > 0:
        kernel = np.ones((erode, erode), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
    binary = keep_components(binary, score, keep=keep, min_area=min_area)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

    soft = binary.astype(np.float32)
    if blur > 0:
        blur = blur + 1 if blur % 2 == 0 else blur
        soft = cv2.GaussianBlur(soft, (blur, blur), 0)
        soft = np.clip(soft, 0.0, 1.0)

    meta = {
        "auto_threshold": auto_threshold,
        "mask_mean": float(soft.mean()),
        "mask_max": float(soft.max()),
    }
    return soft, meta


def alpha_composite(source: np.ndarray, proposal: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    alpha = np.clip(mask[..., None] * strength, 0.0, 1.0)
    out = source.astype(np.float32) * (1.0 - alpha) + proposal.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def seamless_composite(source: np.ndarray, proposal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hard = (mask > 0.15).astype(np.uint8) * 255
    if hard.sum() == 0:
        return source
    ys, xs = np.where(hard > 0)
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
    src_bgr = cv2.cvtColor(proposal, cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
    cloned = cv2.seamlessClone(src_bgr, dst_bgr, hard, center, cv2.NORMAL_CLONE)
    return cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_grid(output: Path, items: list[tuple[str, np.ndarray]]) -> None:
    thumb = 320
    label_h = 38
    pad = 10
    font = load_font(20)
    canvases: list[Image.Image] = []
    for label, array in items:
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
        image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb, thumb + label_h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        draw.text((0, 4), label, fill=(20, 20, 20), font=font)
        canvas.paste(image, ((thumb - image.width) // 2, label_h + (thumb - image.height) // 2))
        canvases.append(canvas)
    grid = Image.new("RGB", (len(canvases) * thumb + (len(canvases) + 1) * pad, thumb + label_h + 2 * pad), (245, 245, 245))
    x = pad
    for canvas in canvases:
        grid.paste(canvas, (x, pad))
        x += thumb + pad
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Composite a local proposal edit back into the source image.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default="proposal_composite")
    parser.add_argument("--roi", default=None, help="Optional normalized x0,y0,x1,y1 ROI used only for mask extraction.")
    parser.add_argument("--threshold", type=float, default=0.22)
    parser.add_argument("--keep-components", type=int, default=2)
    parser.add_argument("--min-area", type=int, default=24)
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--erode", type=int, default=0)
    parser.add_argument("--blur", type=int, default=17)
    parser.add_argument("--dark-bias", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--mode", choices=("alpha", "seamless", "both"), default="both")
    args = parser.parse_args()

    source = load_rgb(args.source)
    proposal = load_rgb(args.proposal, size=(source.shape[1], source.shape[0]))
    mask, meta = build_mask(
        source,
        proposal,
        roi=parse_roi(args.roi),
        threshold=args.threshold,
        keep=args.keep_components,
        min_area=args.min_area,
        dilate=args.dilate,
        erode=args.erode,
        blur=args.blur,
        dark_bias=args.dark_bias,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_rgb = np.repeat((mask * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    save_rgb(args.output_dir / f"{args.name}_mask.png", mask_rgb)

    grid_items = [("source", source), ("proposal", proposal), ("mask", mask_rgb)]
    if args.mode in {"alpha", "both"}:
        alpha = alpha_composite(source, proposal, mask, strength=args.strength)
        save_rgb(args.output_dir / f"{args.name}_alpha.png", alpha)
        grid_items.append(("alpha", alpha))
    if args.mode in {"seamless", "both"}:
        seamless = seamless_composite(source, proposal, mask)
        save_rgb(args.output_dir / f"{args.name}_seamless.png", seamless)
        grid_items.append(("seamless", seamless))

    make_grid(args.output_dir / f"{args.name}_comparison.png", grid_items)
    meta.update(
        {
            "source": str(args.source),
            "proposal": str(args.proposal),
            "roi": args.roi,
            "threshold_floor": args.threshold,
            "keep_components": args.keep_components,
            "min_area": args.min_area,
            "dilate": args.dilate,
            "erode": args.erode,
            "blur": args.blur,
            "dark_bias": args.dark_bias,
            "strength": args.strength,
            "mode": args.mode,
        }
    )
    (args.output_dir / f"{args.name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
