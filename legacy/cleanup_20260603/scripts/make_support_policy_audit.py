from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


TASKS_MAIN = [
    "cat_crown",
    "dog_sunglasses",
    "mug_heart",
    "tshirt_star",
    "tote_leaf",
    "backpack_remove_toy_charm",
    "backpack_replace_patch_blue",
    "cat_replace_bell_heart_tag",
    "dog_replace_tennis_ball_star",
    "rabbit_sunglasses",
    "dog_crown",
]


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = image.copy()
    resample = getattr(Image, "Resampling", Image).LANCZOS
    out.thumbnail(size, resample)
    return out


def tile(path: Path, size: int, label: str) -> Image.Image:
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    if not path.exists():
        draw.text((10, 10), f"missing\n{label}", font=font(14), fill=(130, 0, 0))
        return canvas
    image = Image.open(path).convert("RGB")
    if "mask" in label or "support" in label or "relation" in label or "core" in label:
        image = ImageOps.autocontrast(image)
    image = contain(image, (size, size))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    draw.rectangle((0, 0, size - 1, size - 1), outline=(210, 210, 210))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/Wu_25R8111/rf_h_edit_project"))
    parser.add_argument("--method", default="support_v3_fixed_policyv1_debug")
    parser.add_argument("--seed", default="10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=TASKS_MAIN)
    args = parser.parse_args()

    cols = [
        ("source", "source"),
        ("grounding", "semantic_support.png"),
        ("relation", "operation_v3_relation_map.png"),
        ("selected", "selected_candidate_postprocessed.png"),
        ("core", "M_core.png"),
    ]
    thumb = 190
    left = 270
    header = 42
    row_gap = 12
    col_gap = 12
    width = left + len(cols) * thumb + (len(cols) - 1) * col_gap + 16
    height = header + len(args.tasks) * (thumb + row_gap) + 16
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(17)
    small_font = font(13)

    x0 = left
    for idx, (name, _) in enumerate(cols):
        draw.text((x0 + idx * (thumb + col_gap) + 4, 10), name, font=title_font, fill=(20, 20, 20))

    records: list[dict[str, object]] = []
    y = header
    for task in args.tasks:
        run_dir = args.root / "outputs/pretty_matrix" / task / args.method / f"seed_{args.seed}"
        meta = load_json(run_dir / "metadata.json")
        debug = load_json(run_dir / "masks/operation_v3_debug_metadata.json")
        image_path = Path(str(meta.get("image", "")))
        if not image_path.is_absolute():
            image_path = args.root / image_path
        masks = run_dir / "masks"
        paths = {
            "source": image_path,
            "semantic_support.png": masks / "semantic_support.png",
            "operation_v3_relation_map.png": masks / "operation_v3_relation_map.png",
            "selected_candidate_postprocessed.png": masks / "selected_candidate_postprocessed.png",
            "M_core.png": masks / "M_core.png",
        }
        draw.text((10, y + 6), task, font=title_font, fill=(10, 10, 10))
        selected = str(debug.get("selected_candidate", "?"))
        operation = str(debug.get("operation", meta.get("edit_operation", "?")))
        relation = str(debug.get("relation", meta.get("support_relation", "?")))
        support_area = debug.get("support_area", "?")
        lines = textwrap.wrap(
            f"op={operation} rel={relation} selected={selected} area={support_area}",
            width=32,
        )[:5]
        yy = y + 34
        for line in lines:
            draw.text((10, yy), line, font=small_font, fill=(70, 70, 70))
            yy += 17
        for idx, (label, key) in enumerate(cols):
            canvas.paste(tile(paths[key], thumb, label), (x0 + idx * (thumb + col_gap), y))
        records.append(
            {
                "task": task,
                "run_dir": str(run_dir),
                "image": str(image_path),
                "operation": operation,
                "relation": relation,
                "selected_candidate": selected,
                "support_area": support_area,
                "missing": [label for label, key in cols if not paths[key].exists()],
            }
        )
        y += thumb + row_gap

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
