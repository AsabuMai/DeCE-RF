from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Subtract one or more grayscale masks from a base mask.")
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--subtract", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--subtract-dilate", type=int, default=0)
    parser.add_argument("--subtract-threshold", type=float, default=0.2)
    parser.add_argument("--base-threshold", type=float, default=0.0)
    args = parser.parse_args()

    base_image = Image.open(args.base).convert("L")
    size = base_image.size
    base = np.asarray(base_image, dtype=np.float32) / 255.0
    if args.base_threshold > 0.0:
        base = np.where(base >= args.base_threshold, base, 0.0).astype(np.float32)

    subtract_union = np.zeros_like(base, dtype=np.float32)
    subtract_meta = []
    for path in args.subtract:
        mask = load_mask(path, size=size)
        if args.subtract_threshold > 0.0:
            mask = (mask >= args.subtract_threshold).astype(np.float32)
        if args.subtract_dilate > 1:
            kernel_size = args.subtract_dilate + 1 if args.subtract_dilate % 2 == 0 else args.subtract_dilate
            mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
        subtract_union = np.maximum(subtract_union, mask)
        subtract_meta.append({"path": str(path), "area_gt_0_5": float((mask > 0.5).mean())})

    result = (base * (1.0 - subtract_union)).clip(0.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((result * 255.0).round().astype(np.uint8), mode="L").save(args.output)
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(
                {
                    "base": str(args.base),
                    "subtract": subtract_meta,
                    "output": str(args.output),
                    "base_area_gt_0_5": float((base > 0.5).mean()),
                    "subtract_union_area_gt_0_5": float((subtract_union > 0.5).mean()),
                    "result_area_gt_0_5": float((result > 0.5).mean()),
                    "subtract_dilate": int(args.subtract_dilate),
                    "subtract_threshold": float(args.subtract_threshold),
                    "base_threshold": float(args.base_threshold),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
