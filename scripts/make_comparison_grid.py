from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a labeled horizontal image grid.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("items", nargs="+", help="Grid items as label=path.")
    args = parser.parse_args()

    labels: list[str] = []
    images: list[Image.Image] = []
    for item in args.items:
        label, path = item.split("=", 1)
        labels.append(label)
        images.append(Image.open(path).convert("RGB"))

    thumb_size = 320
    label_h = 42
    pad = 12
    font = load_font(22)
    thumbs = []
    for image in images:
        image.thumbnail((thumb_size, thumb_size), RESAMPLE_LANCZOS)
        canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
        x = (thumb_size - image.width) // 2
        y = (thumb_size - image.height) // 2
        canvas.paste(image, (x, y))
        thumbs.append(canvas)

    width = len(thumbs) * thumb_size + (len(thumbs) + 1) * pad
    height = thumb_size + label_h + 2 * pad
    grid = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(grid)

    for index, (label, thumb) in enumerate(zip(labels, thumbs)):
        x = pad + index * (thumb_size + pad)
        draw.text((x, pad), label, fill=(20, 20, 20), font=font)
        grid.paste(thumb, (x, pad + label_h))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)


if __name__ == "__main__":
    main()
