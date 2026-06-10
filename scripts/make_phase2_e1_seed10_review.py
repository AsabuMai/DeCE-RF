#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/cluster/users/grad/2025/25t8103/project")
RUN_ID = "phase2_e1_seed10_gate_20260605_702647"
RUN_DIR = ROOT / "experiments" / "support_v3_2026-06-02" / RUN_ID
OUT_DIR = RUN_DIR / "visual_review"
OUTPUT_ROOT = ROOT / "outputs" / "pretty_matrix"
THUMB = 180
PAD = 12
LEFT_W = 250
HEADER_H = 62

METHODS = [
    ("base_only", "RF recon."),
    ("direct_target", "Direct target"),
    ("adaptive_full_generic_support", "Generic support"),
    ("support_v3_controller_rmsgap", "DeCE-RF"),
]

TASKS = [
    ("cat_crown", "T1 accessory", "Cat crown"),
    ("dog_bow_tie_phase2", "T1 accessory", "Dog bow tie"),
    ("cat_side_sunglasses_phase2", "T1 accessory", "Side cat sunglasses"),
    ("bowl_apple_inside", "T2 container", "Bowl apple"),
    ("white_bowl_strawberry_phase2", "T2 container", "White bowl strawberry"),
    ("brown_bowl_lemon_phase2", "T2 container", "Brown bowl lemon"),
    ("tshirt_star", "T3 decal", "T-shirt star"),
    ("mug_heart", "T3 decal", "Mug heart"),
    ("tote_leaf", "T3 decal", "Tote leaf"),
    ("red_chair_blue", "T4 recolor", "Room chair blue"),
    ("green_mug_orange_phase2", "T4 recolor", "Green mug orange"),
    ("yellow_vase_blue_phase2", "T4 recolor", "Yellow vase blue"),
    ("pillow_vertical_fabric_strip", "T5 surface", "Pillow vertical strip"),
    ("white_pillow_blue_dots_phase2", "T5 surface", "Pillow dots"),
    ("white_pillow_blue_cross_phase2", "T5 surface", "Pillow cross"),
    ("backpack_remove_toy_charm", "T6 removal", "Backpack charm"),
    ("backpack_remove_silver_keychain_phase2", "T6 removal", "Silver keychain"),
    ("bag_remove_decorative_tag_phase2", "T6 removal", "Bag tag"),
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def fit(path: Path, size: int = THUMB) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def label_cell(text: str, size: int = THUMB) -> Image.Image:
    canvas = Image.new("RGB", (size, size), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, size // 2 - 8), text, fill=(60, 60, 60), font=font(16, True))
    return canvas


def source_for_task(task_id: str) -> Path | None:
    for method, _ in METHODS:
        meta = OUTPUT_ROOT / task_id / method / "seed_10" / "metadata.json"
        if meta.exists():
            data = json.loads(meta.read_text(encoding="utf-8"))
            image = data.get("image") or data.get("source_image")
            if image:
                return Path(image)
    command = OUTPUT_ROOT / task_id / "base_only" / "seed_10" / "command.txt"
    if command.exists():
        parts = command.read_text(encoding="utf-8").split()
        if "--image" in parts:
            return Path(parts[parts.index("--image") + 1])
    return None


def result_for(task_id: str, method: str) -> Path:
    return OUTPUT_ROOT / task_id / method / "seed_10" / "result.png"


def make_sheet(rows: list[tuple[str, str, str]], output: Path, title: str) -> None:
    cols = [("source", "Source"), *METHODS]
    row_h = THUMB + PAD
    width = LEFT_W + PAD + len(cols) * (THUMB + PAD)
    height = HEADER_H + PAD + len(rows) * row_h
    sheet = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 18), title, fill=(15, 15, 15), font=font(22, True))
    for idx, (_, label) in enumerate(cols):
        x = LEFT_W + PAD + idx * (THUMB + PAD)
        draw.text((x, 22), label, fill=(20, 20, 20), font=font(17, True))

    for row_idx, (task_id, category, label) in enumerate(rows):
        y = HEADER_H + PAD + row_idx * row_h
        draw.text((PAD, y + 22), label, fill=(20, 20, 20), font=font(17, True))
        draw.text((PAD, y + 48), category, fill=(75, 75, 75), font=font(13))
        source = source_for_task(task_id)
        cells = [fit(source) if source and source.exists() else label_cell("missing source")]
        for method, _ in METHODS:
            path = result_for(task_id, method)
            cells.append(fit(path) if path.exists() else label_cell("missing"))
        for col_idx, cell in enumerate(cells):
            x = LEFT_W + PAD + col_idx * (THUMB + PAD)
            sheet.paste(cell, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def write_index() -> None:
    path = OUT_DIR / "phase2_e1_seed10_gate_index.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "category", "label", "source_image", *[f"{m}_result" for m, _ in METHODS]])
        for task_id, category, label in TASKS:
            source = source_for_task(task_id)
            writer.writerow([
                task_id,
                category,
                label,
                str(source) if source else "",
                *[str(result_for(task_id, method)) for method, _ in METHODS],
            ])


def write_audit_template() -> None:
    path = OUT_DIR / "phase2_e1_seed10_visual_audit_template.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "task_id",
            "category",
            "label",
            "method",
            "edit_success_1_5",
            "preservation_1_5",
            "locality_1_5",
            "artifact_1_5_lower_better",
            "overall_1_5",
            "gate_decision",
            "notes",
        ])
        for task_id, category, label in TASKS:
            for method, _ in METHODS[1:]:
                writer.writerow([task_id, category, label, method, "", "", "", "", "", "", ""])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_sheet(TASKS, OUT_DIR / "phase2_e1_seed10_all_tasks_grid.png", "Phase2 E1 Seed10 Gate")
    by_category: dict[str, list[tuple[str, str, str]]] = {}
    for row in TASKS:
        by_category.setdefault(row[1], []).append(row)
    for category, rows in by_category.items():
        safe = category.lower().replace(" ", "_")
        make_sheet(rows, OUT_DIR / f"phase2_e1_seed10_{safe}_grid.png", f"Phase2 E1 Seed10 - {category}")
    write_index()
    write_audit_template()
    completion = {
        "all_tasks_grid": str(OUT_DIR / "phase2_e1_seed10_all_tasks_grid.png"),
        "index_csv": str(OUT_DIR / "phase2_e1_seed10_gate_index.csv"),
        "audit_template_csv": str(OUT_DIR / "phase2_e1_seed10_visual_audit_template.csv"),
        "task_count": len(TASKS),
        "method_count": len(METHODS),
    }
    (OUT_DIR / "phase2_e1_seed10_review_package.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
