#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edit_preprocess import preprocess_source_image


TASKS = [
    ("laptop_remove_sticker", "surface", "laptop"),
    ("fridge_remove_yellow_magnet", "surface", "fridge yellow"),
    ("fridge_remove_peach_magnet", "surface", "fridge peach"),
    ("whiteboard_remove_yellow_letter", "surface", "whiteboard"),
    ("backpack_remove_toy_charm", "hard", "backpack"),
    ("dog_remove_tennis_ball", "hard", "dog"),
]

RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_source_rgb(path: Path, max_image_size: int, size: tuple[int, int]) -> np.ndarray:
    image = preprocess_source_image(path, max_image_size=max_image_size)
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(mask, dtype=np.uint8) > 0


def to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((image * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def gray_grad(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((image * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0


def safe_mean(values: np.ndarray, fallback: float = 0.0) -> float:
    if values.size == 0:
        return fallback
    return float(np.mean(values))


def exp_score(value: float, scale: float) -> float:
    return float(np.exp(-max(0.0, value) / max(scale, 1e-6)))


def mean_lab_delta(a_lab: np.ndarray, b_lab: np.ndarray, region: np.ndarray) -> float:
    if not bool(region.any()):
        return 999.0
    diff = a_lab[region] - b_lab[region]
    return float(np.sqrt((diff * diff).sum(axis=1)).mean())


def boundary_score(source: np.ndarray, prior: np.ndarray, mask: np.ndarray, ring_px: int) -> tuple[float, dict[str, float]]:
    inner = mask & ~erode(mask, ring_px)
    outer = dilate(mask, ring_px) & ~mask
    if not bool(inner.any()) or not bool(outer.any()):
        return 0.0, {"boundary_color_delta": 999.0, "boundary_grad_delta": 999.0}

    prior_lab = to_lab(prior)
    source_lab = to_lab(source)
    prior_grad = gray_grad(prior)
    source_grad = gray_grad(source)

    inner_color = prior_lab[inner].mean(axis=0)
    outer_color = source_lab[outer].mean(axis=0)
    color_delta = float(np.sqrt(((inner_color - outer_color) ** 2).sum()))
    grad_delta = abs(safe_mean(prior_grad[inner]) - safe_mean(source_grad[outer]))
    inner_texture_std = float(prior_lab[inner].std(axis=0).mean() / 255.0)
    inner_grad_mean = safe_mean(prior_grad[inner])

    color_score = exp_score(color_delta, 32.0)
    grad_score = exp_score(grad_delta, 0.16)
    inner_texture_score = exp_score(inner_texture_std, 0.07)
    inner_grad_score = exp_score(inner_grad_mean, 0.12)
    score = float((color_score * grad_score * inner_texture_score * inner_grad_score) ** 0.25)
    return score, {
        "boundary_color_delta": color_delta,
        "boundary_grad_delta": grad_delta,
        "boundary_inner_texture_std": inner_texture_std,
        "boundary_inner_grad_mean": inner_grad_mean,
        "boundary_color_score": color_score,
        "boundary_grad_score": grad_score,
        "boundary_inner_texture_score": inner_texture_score,
        "boundary_inner_grad_score": inner_grad_score,
    }


def agreement_score(telea: np.ndarray, ns: np.ndarray, mask: np.ndarray) -> tuple[float, dict[str, float]]:
    if not bool(mask.any()):
        return 0.0, {"agreement_lab_delta": 999.0, "agreement_grad_delta": 999.0}
    telea_lab = to_lab(telea)
    ns_lab = to_lab(ns)
    telea_grad = gray_grad(telea)
    ns_grad = gray_grad(ns)

    lab_delta = mean_lab_delta(telea_lab, ns_lab, mask)
    grad_delta = safe_mean(np.abs(telea_grad[mask] - ns_grad[mask]))
    color_score = exp_score(lab_delta, 12.0)
    grad_score = exp_score(grad_delta, 0.10)
    score = float(np.sqrt(color_score * grad_score))
    return score, {
        "agreement_lab_delta": lab_delta,
        "agreement_grad_delta": grad_delta,
        "agreement_color_score": color_score,
        "agreement_grad_score": grad_score,
    }


def host_score(source: np.ndarray, mask: np.ndarray, ring_px: int) -> tuple[float, dict[str, float]]:
    ring = dilate(mask, ring_px * 2) & ~mask
    if not bool(ring.any()):
        return 0.0, {"host_texture_std": 999.0, "host_edge_mean": 999.0, "host_edge_density": 999.0}
    lab = to_lab(source)
    grad = gray_grad(source)
    texture_std = float(lab[ring].std(axis=0).mean() / 255.0)
    edge_mean = safe_mean(grad[ring])
    edge_density = float((grad[ring] > 0.20).mean())
    texture_score = exp_score(texture_std, 0.08)
    edge_score = exp_score(edge_mean, 0.18)
    edge_density_score = exp_score(edge_density, 0.22)
    score = float((texture_score * edge_score * edge_density_score) ** (1.0 / 3.0))
    return score, {
        "host_texture_std": texture_std,
        "host_edge_mean": edge_mean,
        "host_edge_density": edge_density,
        "host_texture_score": texture_score,
        "host_edge_score": edge_score,
        "host_edge_density_score": edge_density_score,
    }


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), RESAMPLE_LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_grid(rows: list[dict[str, object]], root: Path, output: Path, seed: int, thumb: int) -> None:
    columns = [("source", "source"), ("telea", "Telea"), ("ns", "NS"), ("clean_delta", "clean_delta")]
    left_w = 230
    pad = 12
    header_h = 62
    row_h = thumb + pad
    width = left_w + len(columns) * (thumb + pad) + pad
    height = header_h + len(rows) * row_h + pad
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(18, True)
    row_font = load_font(17, True)
    small_font = load_font(14)

    for col_index, (_, label) in enumerate(columns):
        x = left_w + pad + col_index * (thumb + pad)
        draw.text((x, 20), label, fill=(20, 20, 20), font=header_font)

    for row_index, row in enumerate(rows):
        task = str(row["task"])
        y = header_h + row_index * row_h
        draw.text((pad, y + 16), str(row["label"]), fill=(20, 20, 20), font=row_font)
        draw.text((pad, y + 42), f"R={row['R']:.2f} B={row['R_boundary']:.2f} A={row['R_agreement']:.2f} H={row['R_host']:.2f}", fill=(70, 70, 70), font=small_font)

        default = root / "outputs" / "pretty_matrix" / task / "support_v3_controller_rmsgap" / f"seed_{seed}"
        source = Path(json.loads((default / "metadata.json").read_text(encoding="utf-8"))["image"])
        paths = {
            "source": source,
            "telea": root / "outputs" / "pretty_matrix" / task / "same_support_inpaint_telea" / f"seed_{seed}" / "result.png",
            "ns": root / "outputs" / "pretty_matrix" / task / "same_support_inpaint_ns" / f"seed_{seed}" / "result.png",
            "clean_delta": root / "outputs" / "pretty_matrix" / task / "support_v3_controller_rmsgap_completion_clean_delta" / f"seed_{seed}" / "result.png",
        }
        for col_index, (key, _) in enumerate(columns):
            x = left_w + pad + col_index * (thumb + pad)
            canvas.paste(fit_image(paths[key], thumb), (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose classical completion prior reliability for removal tasks.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--tasks", default=" ".join(task for task, _, _ in TASKS))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability"))
    parser.add_argument("--ring-px", type=int, default=8)
    parser.add_argument("--thumb", type=int, default=180)
    args = parser.parse_args()

    task_meta = {task: (group, label) for task, group, label in TASKS}
    tasks = [item for item in args.tasks.replace(",", " ").split() if item]
    rows: list[dict[str, object]] = []
    for task in tasks:
        group, label = task_meta.get(task, ("custom", task))
        default = args.root / "outputs" / "pretty_matrix" / task / "support_v3_controller_rmsgap" / f"seed_{args.seed}"
        telea_dir = args.root / "outputs" / "pretty_matrix" / task / "same_support_inpaint_telea" / f"seed_{args.seed}"
        ns_dir = args.root / "outputs" / "pretty_matrix" / task / "same_support_inpaint_ns" / f"seed_{args.seed}"
        meta = json.loads((default / "metadata.json").read_text(encoding="utf-8"))
        telea = load_rgb(telea_dir / "result.png")
        ns = load_rgb(ns_dir / "result.png")
        source = load_source_rgb(Path(meta["image"]), int(meta.get("max_image_size") or 512), (telea.shape[1], telea.shape[0]))
        mask = load_mask(telea_dir / "masks" / "same_support_inpaint_mask.png", (source.shape[1], source.shape[0]))

        r_boundary, boundary_details = boundary_score(source, telea, mask, args.ring_px)
        r_agreement, agreement_details = agreement_score(telea, ns, mask)
        r_host, host_details = host_score(source, mask, args.ring_px)
        r_total = float(r_boundary * r_agreement * r_host)
        row: dict[str, object] = {
            "task": task,
            "group": group,
            "label": label,
            "seed": args.seed,
            "mask_area_ratio": float(mask.mean()),
            "R_boundary": r_boundary,
            "R_agreement": r_agreement,
            "R_host": r_host,
            "R": r_total,
        }
        row.update(boundary_details)
        row.update(agreement_details)
        row.update(host_details)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"completion_prior_reliability_seed{args.seed}.json"
    csv_path = args.output_dir / f"completion_prior_reliability_seed{args.seed}.csv"
    grid_path = args.output_dir / f"completion_prior_reliability_seed{args.seed}_grid.png"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    make_grid(rows, args.root, grid_path, args.seed, args.thumb)
    print(csv_path)
    print(json_path)
    print(grid_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
