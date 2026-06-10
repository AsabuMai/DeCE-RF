"""Apply the host-color chroma guard to a semantic support mask in place.

Used for dark hosts where SAM over-segmentation can include differently
colored background (e.g. rattan next to a grey pillow); the guard removes
chroma outliers so downstream M_preserve covers them again.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_decal_reference import clip_mask_to_host_color  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to overwriting --mask.")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    guarded = clip_mask_to_host_color(image, mask)
    out = args.output or args.mask
    guarded.save(out)
    print(f"[chroma-guard] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
