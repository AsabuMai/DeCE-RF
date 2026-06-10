#!/usr/bin/env python3
"""Apply a fixed boundary refinement to surface recolor masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--erode-kernel", type=int, default=3)
    parser.add_argument("--erode-iterations", type=int, default=1)
    parser.add_argument("--dilate-kernel", type=int, default=0)
    parser.add_argument("--dilate-iterations", type=int, default=0)
    parser.add_argument("--blur-kernel", type=int, default=3)
    args = parser.parse_args()

    mask_image = Image.open(args.mask).convert("L")
    mask = np.asarray(mask_image, dtype=np.float32) / 255.0
    binary = (mask >= float(args.threshold)).astype(np.uint8)

    if args.erode_kernel > 1 and args.erode_iterations > 0:
        kernel_size = int(args.erode_kernel)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.erode(binary, kernel, iterations=int(args.erode_iterations))

    if args.dilate_kernel > 1 and args.dilate_iterations > 0:
        kernel_size = int(args.dilate_kernel)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=int(args.dilate_iterations))

    refined = binary.astype(np.float32)
    if args.blur_kernel > 1:
        blur_size = int(args.blur_kernel)
        if blur_size % 2 == 0:
            blur_size += 1
        refined = cv2.GaussianBlur(refined, (blur_size, blur_size), 0)
        if float(refined.max()) > 1e-6:
            refined = refined / float(refined.max())

    refined_u8 = (np.clip(refined, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(refined_u8, mode="L").save(args.output)

    if args.metadata_output is not None:
        payload = {
            "mask": str(args.mask),
            "output": str(args.output),
            "threshold": float(args.threshold),
            "erode_kernel": int(args.erode_kernel),
            "erode_iterations": int(args.erode_iterations),
            "dilate_kernel": int(args.dilate_kernel),
            "dilate_iterations": int(args.dilate_iterations),
            "blur_kernel": int(args.blur_kernel),
            "input_area_gt_0_5": float((mask >= 0.5).mean()),
            "output_area_gt_0_5": float((refined >= 0.5).mean()),
        }
        args.metadata_output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
