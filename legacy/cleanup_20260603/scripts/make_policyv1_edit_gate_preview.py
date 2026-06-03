from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TASKS = [
    "tshirt_star",
    "tote_leaf",
    "backpack_remove_toy_charm",
    "backpack_replace_patch_blue",
    "cat_replace_bell_heart_tag",
    "dog_replace_tennis_ball_star",
    "rabbit_sunglasses",
]

DEFAULT_METHODS = [
    ("fixed", "support_v3_fixed"),
    ("rmsgap", "support_v3_controller_rmsgap"),
    ("opfield", "support_v3_controller_rmsgap_opfield"),
    ("M22", "support_v3_core_target_transport"),
]


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


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
    image = contain(Image.open(path).convert("RGB"), (size, size))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    draw.rectangle((0, 0, size - 1, size - 1), outline=(210, 210, 210))
    return canvas


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_methods(values: list[str]) -> list[tuple[str, str]]:
    methods: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--methods entries must be label=method, got {value!r}")
        label, method = value.split("=", 1)
        label = label.strip()
        method = method.strip()
        if not label or not method:
            raise ValueError(f"--methods entries must be label=method, got {value!r}")
        methods.append((label, method))
    return methods


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/Wu_25R8111/rf_h_edit_project"))
    parser.add_argument("--seed", default="10")
    parser.add_argument("--scales", nargs="*", default=["100", "150"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=TASKS)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[f"{label}={method}" for label, method in DEFAULT_METHODS],
    )
    parser.add_argument(
        "--exact-method-dirs",
        action="store_true",
        default=False,
        help="Treat method values as complete output directory names instead of appending _policyv1_edit{scale}.",
    )
    args = parser.parse_args()
    methods = parse_methods(args.methods)

    thumb = 190
    left = 250
    header = 42
    row_gap = 12
    col_gap = 12
    title_font = font(17)
    small_font = font(13)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scale in args.scales:
        if args.exact_method_dirs:
            cols = [("source", "source")] + methods
        else:
            cols = [("source", "source")] + [(name, f"{method}_policyv1_edit{scale}") for name, method in methods]
        width = left + len(cols) * thumb + (len(cols) - 1) * col_gap + 16
        height = header + len(args.tasks) * (thumb + row_gap) + 16
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        x0 = left
        for idx, (name, _) in enumerate(cols):
            draw.text((x0 + idx * (thumb + col_gap) + 4, 10), name, font=title_font, fill=(20, 20, 20))

        records: list[dict[str, object]] = []
        y = header
        for task in args.tasks:
            first_method_dir = methods[0][1] if args.exact_method_dirs else f"{methods[0][1]}_policyv1_edit{scale}"
            first_meta = args.root / "outputs/pretty_matrix" / task / first_method_dir / f"seed_{args.seed}" / "metadata.json"
            meta = load_json(first_meta)
            image_path = Path(str(meta.get("image", "")))
            if not image_path.is_absolute():
                image_path = args.root / image_path
            target = str(meta.get("target_prompt", ""))
            draw.text((10, y + 5), task, font=title_font, fill=(10, 10, 10))
            yy = y + 32
            for line in textwrap.wrap(target, width=30)[:5]:
                draw.text((10, yy), line, font=small_font, fill=(70, 70, 70))
                yy += 17

            missing: list[str] = []
            for idx, (label, key) in enumerate(cols):
                if key == "source":
                    path = image_path
                else:
                    path = args.root / "outputs/pretty_matrix" / task / key / f"seed_{args.seed}" / "result.png"
                if not path.exists():
                    missing.append(label)
                canvas.paste(tile(path, thumb, label), (x0 + idx * (thumb + col_gap), y))
            records.append({"task": task, "scale": scale, "missing": missing})
            y += thumb + row_gap

        out = args.output_dir / f"policyv1_edit_gate_scale{scale}.png"
        report = args.output_dir / f"policyv1_edit_gate_scale{scale}.json"
        canvas.save(out)
        report.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(out)
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
