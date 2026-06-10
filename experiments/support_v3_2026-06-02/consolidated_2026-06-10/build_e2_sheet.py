"""Build E2 T1-T4 seed-10 RF-baseline comparison sheet. Light PIL-only work."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/cluster/users/grad/2025/25t8103/project")
PACK = ROOT / "experiments/support_v3_2026-06-02"
OUT_DIR = PACK / "consolidated_2026-06-10"

GROUPS = [
    ("T1", ["cat_crown", "dog_bow_tie_phase2", "dog_front_sunglasses_phase2"]),
    ("T2", ["bowl_apple_inside", "white_bowl_orange_tabletop_phase2", "brown_bowl_lemon_phase2"]),
    ("T3", ["tshirt_star", "mug_heart", "tote_leaf"]),
    ("T4", ["red_office_chair_to_blue_office_chair", "green_mug_orange_phase2", "yellow_vase_blue_phase2"]),
]
BASELINES = ["flowedit", "flowalign", "splitflow", "rf_solver_edit", "fireflow", "reflex"]
CELL = 256
LABEL_W = 230
HEADER_H = 34
SEED = "10"

font = ImageFont.load_default()
manifest = {}
with open(PACK / "e2_t1_t4_baseline_matrix_manifest.csv") as f:
    for r in csv.DictReader(f):
        if r["seed"] == SEED:
            manifest[(r["baseline"], r["task"])] = ROOT / r["result"]


def cell_image(path: Path | None) -> Image.Image:
    if path and path.is_file():
        img = Image.open(path).convert("RGB")
        img.thumbnail((CELL, CELL))
        canvas = Image.new("RGB", (CELL, CELL), (240, 240, 240))
        canvas.paste(img, ((CELL - img.width) // 2, (CELL - img.height) // 2))
        return canvas
    canvas = Image.new("RGB", (CELL, CELL), (60, 60, 60))
    ImageDraw.Draw(canvas).text((10, CELL // 2), "missing", fill=(255, 255, 255), font=font)
    return canvas


tasks = [t for _, names in GROUPS for t in names]
group_of = {t: g for g, names in GROUPS for t in names}
cols = 1 + len(BASELINES) + 1  # source + baselines + DeCE-RF
sheet = Image.new("RGB", (LABEL_W + cols * CELL, HEADER_H + len(tasks) * CELL), (255, 255, 255))
draw = ImageDraw.Draw(sheet)
for i, name in enumerate(["source"] + BASELINES + ["DeCE-RF"]):
    draw.text((LABEL_W + i * CELL + 8, 10), name, fill=(0, 0, 0), font=font)

for row, task in enumerate(tasks):
    y = HEADER_H + row * CELL
    draw.text((6, y + 8), f"{group_of[task]} {task[:32]}", fill=(0, 0, 0), font=font)
    src_path = None
    meta = ROOT / "outputs/pretty_matrix" / task / "base_only" / f"seed_{SEED}" / "metadata.json"
    if meta.is_file():
        m = json.loads(meta.read_text())
        src = m.get("source_image") or m.get("image")
        if src:
            src_path = Path(src)
    sheet.paste(cell_image(src_path), (LABEL_W, y))
    for j, b in enumerate(BASELINES):
        sheet.paste(cell_image(manifest.get((b, task))), (LABEL_W + (j + 1) * CELL, y))
    dece = ROOT / "outputs/pretty_matrix" / task / "support_v3_controller_rmsgap" / f"seed_{SEED}" / "result.png"
    sheet.paste(cell_image(dece), (LABEL_W + (len(BASELINES) + 1) * CELL, y))

out = OUT_DIR / "e2_t1_t4_seed10_rf_baseline_sheet.png"
sheet.save(out)
print("sheet:", out, sheet.size)
