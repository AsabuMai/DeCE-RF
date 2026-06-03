#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edit_preprocess import preprocess_source_image


MASK_CANDIDATES = (
    "masks/operation_v3_edit_mask.png",
    "masks/subject_final.png",
    "masks/selected_candidate_postprocessed.png",
    "masks/semantic_support.png",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_mask(run_dir: Path) -> Path:
    for rel in MASK_CANDIDATES:
        path = run_dir / rel
        if path.exists():
            return path
    raise FileNotFoundError(f"No support mask found under {run_dir}")


def current_git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def make_mask(mask_path: Path, size: tuple[int, int], dilate: int, blur: float, threshold: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.BILINEAR)
    hard = mask.point(lambda value: 255 if value >= threshold else 0)
    if dilate > 0:
        hard = hard.filter(ImageFilter.MaxFilter(2 * dilate + 1))
    if blur > 0:
        hard = hard.filter(ImageFilter.GaussianBlur(radius=blur))
        hard = hard.point(lambda value: 255 if value >= threshold else 0)
    return np.asarray(hard, dtype=np.uint8)


def run_inpaint(image: Image.Image, mask: np.ndarray, method: str, radius: float) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    result = cv2.inpaint(bgr, mask, radius, flag)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def process_task(
    root: Path,
    task: str,
    seed: str,
    source_method: str,
    output_method: str,
    inpaint_method: str,
    radius: float,
    dilate: int,
    blur: float,
    threshold: int,
) -> None:
    source_run = root / "outputs" / "pretty_matrix" / task / source_method / f"seed_{seed}"
    metadata = load_json(source_run / "metadata.json")
    mask_path = find_mask(source_run)
    max_image_size = int(metadata.get("max_image_size") or 512)
    source_image = preprocess_source_image(metadata["image"], max_image_size=max_image_size)
    mask = make_mask(mask_path, source_image.size, dilate=dilate, blur=blur, threshold=threshold)
    result = run_inpaint(source_image, mask, method=inpaint_method, radius=radius)

    out_dir = root / "outputs" / "pretty_matrix" / task / output_method / f"seed_{seed}"
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    result.save(out_dir / "result.png")
    Image.fromarray(mask, mode="L").save(mask_dir / "same_support_inpaint_mask.png")

    out_meta = {
        "image": metadata["image"],
        "source_prompt": metadata.get("source_prompt"),
        "target_prompt": metadata.get("target_prompt"),
        "output": str(out_dir / "result.png"),
        "seed": seed,
        "method": output_method,
        "diagnostic": "same_support_classical_inpaint",
        "inpaint_method": inpaint_method,
        "inpaint_radius": radius,
        "inpaint_dilate": dilate,
        "inpaint_blur": blur,
        "inpaint_threshold": threshold,
        "support_source_run": str(source_run),
        "support_mask": str(mask_path),
        "max_image_size": max_image_size,
        "git_commit": current_git_commit(root),
    }
    (out_dir / "metadata.json").write_text(json.dumps(out_meta, indent=2) + "\n", encoding="utf-8")
    stats = [
        {
            "diagnostic": "same_support_classical_inpaint",
            "inpaint_method": inpaint_method,
            "mask_area_ratio": float((mask > 0).mean()),
        }
    ]
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    command = (
        f"python scripts/same_support_inpaint_baseline.py --task {task} --seed {seed} "
        f"--source-method {source_method} --method {inpaint_method} --radius {radius} "
        f"--dilate {dilate} --blur {blur} --threshold {threshold}\n"
    )
    (out_dir / "command.txt").write_text(command, encoding="utf-8")
    print(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run same-support classical inpainting removal diagnostics.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--tasks", default="backpack_remove_toy_charm laptop_remove_sticker dog_remove_tennis_ball")
    parser.add_argument("--seed", default="10")
    parser.add_argument("--source-method", default="support_v3_controller_rmsgap")
    parser.add_argument("--method", choices=("telea", "ns"), default="telea")
    parser.add_argument("--output-method", default=None)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--dilate", type=int, default=3)
    parser.add_argument("--blur", type=float, default=0.0)
    parser.add_argument("--threshold", type=int, default=45)
    args = parser.parse_args()

    output_method = args.output_method or f"same_support_inpaint_{args.method}"
    for task in [item.strip() for item in args.tasks.replace(",", " ").split() if item.strip()]:
        process_task(
            root=args.root,
            task=task,
            seed=args.seed,
            source_method=args.source_method,
            output_method=output_method,
            inpaint_method=args.method,
            radius=args.radius,
            dilate=args.dilate,
            blur=args.blur,
            threshold=args.threshold,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
