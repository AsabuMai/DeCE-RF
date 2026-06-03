#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TASKS = [
    ("laptop_remove_sticker", "laptop"),
    ("fridge_remove_yellow_magnet", "fridge yellow"),
    ("fridge_remove_peach_magnet", "fridge peach"),
    ("whiteboard_remove_yellow_letter", "whiteboard"),
    ("backpack_remove_toy_charm", "backpack"),
    ("dog_remove_tennis_ball", "dog"),
]

DEFAULT_GATED_METHOD = "support_v3_controller_rmsgap_completion_clean_delta_gated_highconf"

BASE_COLUMNS = [
    ("source", "source"),
    ("support_v3_controller_rmsgap", "M0 default"),
    ("support_v3_controller_rmsgap_completion_clean_delta", "M1 clean_delta"),
]

RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_reliability(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["task"]: float(row["R"]) for row in csv.DictReader(handle)}


def load_gate(protocol: Path) -> dict[tuple[str, int], dict]:
    if not protocol.exists():
        return {}
    data = json.loads(protocol.read_text(encoding="utf-8"))
    out = {}
    for row in data.get("per_task_gate", []):
        out[(str(row["task"]), int(row["seed"]))] = row
    return out


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), RESAMPLE_LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def source_path(root: Path, task: str, seed: int) -> Path | None:
    metadata_path = root / task / "support_v3_controller_rmsgap" / f"seed_{seed}" / "metadata.json"
    if not metadata_path.exists():
        return None
    image = json.loads(metadata_path.read_text(encoding="utf-8")).get("image")
    return Path(image) if image else None


def result_path(root: Path, task: str, method: str, seed: int) -> Path | None:
    if method == "source":
        return source_path(root, task, seed)
    return root / task / method / f"seed_{seed}" / "result.png"


def missing_cell(size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    font = load_font(16)
    label = "missing"
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.text(((size - (bbox[2] - bbox[0])) // 2, (size - (bbox[3] - bbox[1])) // 2), label, fill=(90, 90, 90), font=font)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cross-seed M0/M1/M2 gated clean-delta grid.")
    parser.add_argument("--root", type=Path, default=Path("outputs/pretty_matrix"))
    parser.add_argument("--seeds", default="10 11 12")
    parser.add_argument("--thumb", type=int, default=160)
    parser.add_argument("--gated-method", default=DEFAULT_GATED_METHOD)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_highconf_seeds10_11_12_protocol.json"))
    parser.add_argument("--reliability-dir", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability"))
    parser.add_argument("--output", type=Path, default=Path("experiments/support_v3_2026-05-11/visual_gates/removal_completion_clean_delta_gated_highconf_seeds10_11_12_grid.png"))
    args = parser.parse_args()
    columns = BASE_COLUMNS + [(args.gated_method, "M2 highconf")]

    seeds = [int(item) for item in args.seeds.replace(",", " ").split() if item]
    gates = load_gate(args.protocol)
    reliability = {
        seed: load_reliability(args.reliability_dir / f"completion_prior_reliability_seed{seed}.csv")
        for seed in seeds
    }

    left_w = 250
    header_h = 58
    pad = 10
    row_h = args.thumb + pad
    rows = [(task, label, seed) for task, label in TASKS for seed in seeds]
    width = left_w + len(columns) * (args.thumb + pad) + pad
    height = header_h + len(rows) * row_h + pad
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(17, True)
    row_font = load_font(15, True)
    small_font = load_font(13)

    for col_index, (_, label) in enumerate(columns):
        x = left_w + pad + col_index * (args.thumb + pad)
        draw.text((x, 20), label, fill=(24, 24, 24), font=header_font)

    for row_index, (task, label, seed) in enumerate(rows):
        y = header_h + row_index * row_h
        gate = gates.get((task, seed), gates.get((task, 10), {}))
        factor = float(gate.get("gate_factor", 0.0))
        score = reliability.get(seed, {}).get(task, reliability.get(10, {}).get(task, 0.0))
        draw.text((pad, y + 16), f"{label}  seed {seed}", fill=(24, 24, 24), font=row_font)
        draw.text((pad, y + 38), f"R={score:.2f} gate={factor:.1f}", fill=(70, 70, 70), font=small_font)
        for col_index, (method, _) in enumerate(columns):
            path = result_path(args.root, task, method, seed)
            cell = fit_image(path, args.thumb) if path is not None and path.exists() else missing_cell(args.thumb)
            x = left_w + pad + col_index * (args.thumb + pad)
            canvas.paste(cell, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
