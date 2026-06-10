from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TASKS = [
    "cat_crown",
    "bowl_apple_inside",
    "tshirt_star",
    "red_chair_blue",
    "pillow_vertical_fabric_strip",
    "backpack_remove_toy_charm",
]
METHODS = ["fireflow", "rf_solver_edit", "reflex"]
SEEDS = ["10", "11", "12"]
CANVAS = 512


def fit_to_canvas(path: Path, output: Path, mode: str = "RGB") -> tuple[str, str, str]:
    image = Image.open(path).convert(mode)
    original_size = image.size
    scale = min(CANVAS / image.width, CANVAS / image.height)
    fitted_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resample = Image.Resampling.BICUBIC if mode == "RGB" else Image.Resampling.BILINEAR
    fitted = image.resize(fitted_size, resample)
    background = (0, 0, 0) if mode == "RGB" else 0
    canvas = Image.new(mode, (CANVAS, CANVAS), background)
    offset = ((CANVAS - fitted_size[0]) // 2, (CANVAS - fitted_size[1]) // 2)
    canvas.paste(fitted, offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return (
        f"{original_size[0]}x{original_size[1]}",
        f"{fitted_size[0]}x{fitted_size[1]}",
        f"{offset[0]},{offset[1]}",
    )


def load_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows: list[dict[str, str]], out_path: Path) -> None:
    fieldnames = [
        "group",
        "task",
        "method",
        "seed",
        "source_path",
        "normalized_path",
        "original_size",
        "fitted_size",
        "offset",
        "backbone",
        "normalization_policy",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_cell(image: Image.Image, text: str) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    box_h = 34
    draw.rectangle((0, 0, canvas.width, box_h), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    return canvas


def make_seed_grid(seed: str, normalized_root: Path, out_path: Path) -> None:
    cols = ["source", *METHODS]
    cell_w = CANVAS
    cell_h = CANVAS
    header_h = 44
    grid = Image.new("RGB", (cell_w * len(cols), header_h + cell_h * len(TASKS)), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for c, name in enumerate(cols):
        draw.rectangle((c * cell_w, 0, (c + 1) * cell_w, header_h), fill=(32, 32, 32))
        draw.text((c * cell_w + 10, 10), name, fill=(255, 255, 255), font=font)

    for r, task in enumerate(TASKS):
        y = header_h + r * cell_h
        source = Image.open(normalized_root / "sources" / f"{task}.png").convert("RGB")
        grid.paste(label_cell(source, task), (0, y))
        for c, method in enumerate(METHODS, start=1):
            path = normalized_root / "e2_native_flux" / method / task / f"seed_{seed}.png"
            if path.exists():
                image = Image.open(path).convert("RGB")
            else:
                image = Image.new("RGB", (CANVAS, CANVAS), (80, 0, 0))
                ImageDraw.Draw(image).text((20, 20), "MISSING", fill=(255, 255, 255), font=small_font)
            grid.paste(label_cell(image, method), (c * cell_w, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main() -> int:
    root = Path.cwd()
    experiment = root / "experiments" / "support_v3_2026-06-02"
    manifest = experiment / "e2_strict_rf_baseline_manifest.csv"
    normalized_root = experiment / "normalized_512"
    out_manifest = experiment / "e2_native_flux_normalized_512_manifest.csv"
    rows = load_rows(manifest)

    manifest_rows: list[dict[str, str]] = []
    for task in TASKS:
        src = normalized_root / "sources" / f"{task}.png"
        if src.exists():
            manifest_rows.append(
                {
                    "group": "source",
                    "task": task,
                    "method": "",
                    "seed": "",
                    "source_path": str(src),
                    "normalized_path": str(src),
                    "original_size": "512x512",
                    "fitted_size": "512x512",
                    "offset": "0,0",
                    "backbone": "",
                    "normalization_policy": "existing normalized_512 source",
                }
            )

    by_key = {
        (row["baseline"], row["task"], row["seed"]): row
        for row in rows
        if row.get("baseline") in METHODS and row.get("seed") in SEEDS
    }
    for method in METHODS:
        for task in TASKS:
            for seed in SEEDS:
                row = by_key[(method, task, seed)]
                if row.get("status") != "complete":
                    raise RuntimeError(f"{method}/{task}/seed_{seed} is not complete")
                source = root / row["result_image"]
                output = normalized_root / "e2_native_flux" / method / task / f"seed_{seed}.png"
                original_size, fitted_size, offset = fit_to_canvas(source, output)
                metrics_dir = root / "outputs" / "e2_native_flux_contextual" / task / method / f"seed_{seed}"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output, metrics_dir / "result.png")
                if row.get("command"):
                    shutil.copy2(root / row["command"], metrics_dir / "command.txt")
                else:
                    (metrics_dir / "command.txt").write_text("command unavailable\n", encoding="utf-8")
                (metrics_dir / "stats.json").write_text(
                    json.dumps(
                        {
                            "steps": [],
                            "note": "Native FLUX contextual baseline; no DeCE controller trajectory stats.",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (metrics_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "baseline": method,
                            "method": method,
                            "task": task,
                            "seed": int(seed),
                            "image": str(normalized_root / "sources" / f"{task}.png"),
                            "source_prompt": row["source_prompt"],
                            "target_prompt": row["target_prompt"],
                            "result_image": str(output),
                            "original_result_image": row["result_image"],
                            "backbone": "FLUX.1-dev",
                            "native_backbone": "FLUX.1-dev",
                            "resolution": [CANVAS, CANVAS],
                            "normalization_policy": (
                                "Metrics-ready result is fit to 512x512 canvas with aspect ratio preserved "
                                "and black letterbox; original native output is retained in outputs/baselines."
                            ),
                            "matched_conditions": row.get("matched_conditions", ""),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                manifest_rows.append(
                    {
                        "group": "e2_native_flux_result",
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "source_path": row["result_image"],
                        "normalized_path": str(output.relative_to(root)),
                        "original_size": original_size,
                        "fitted_size": fitted_size,
                        "offset": offset,
                        "backbone": "FLUX.1-dev",
                        "normalization_policy": "fit original result to 512x512 canvas; preserve aspect ratio; black letterbox; original output retained",
                    }
                )

    write_manifest(manifest_rows, out_manifest)
    grid_dir = experiment / "visual_audit" / "e2_native_flux_grids"
    for seed in SEEDS:
        make_seed_grid(seed, normalized_root, grid_dir / f"e2_native_flux_seed{seed}_grid.png")
    print(f"wrote {out_manifest}")
    print(f"wrote grids to {grid_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
