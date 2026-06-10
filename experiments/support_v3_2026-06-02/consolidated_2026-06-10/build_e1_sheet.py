"""Build E1 phase2 T1-T4 seed-10 comparison sheet and aggregate metrics.

Light CPU-only work (PIL paste + csv means); safe on the master node.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/cluster/users/grad/2025/25t8103/project")
PACK = ROOT / "experiments/support_v3_2026-06-02"
OUT_DIR = PACK / "consolidated_2026-06-10"
OUT_DIR.mkdir(exist_ok=True)

GROUPS = [
    ("T1 attached accessory", ["cat_crown", "dog_bow_tie_phase2", "dog_front_sunglasses_phase2"]),
    ("T2 container insertion", ["bowl_apple_inside", "white_bowl_orange_tabletop_phase2", "brown_bowl_lemon_phase2"]),
    ("T3 surface decal", ["tshirt_star", "mug_heart", "tote_leaf"]),
    ("T4 local recolor", ["red_office_chair_to_blue_office_chair", "green_mug_orange_phase2", "yellow_vase_blue_phase2"]),
]
METHODS = [
    ("base_only", "base_only (recon)"),
    ("direct_target", "direct_target"),
    ("adaptive_full_generic_support", "generic support"),
    ("support_v3_controller_rmsgap", "DeCE-RF"),
]
CELL = 256
LABEL_W = 230
HEADER_H = 34
SEED = 10

font = ImageFont.load_default()


def cell_image(path: Path) -> Image.Image:
    if path.is_file():
        img = Image.open(path).convert("RGB")
        img.thumbnail((CELL, CELL))
        canvas = Image.new("RGB", (CELL, CELL), (240, 240, 240))
        canvas.paste(img, ((CELL - img.width) // 2, (CELL - img.height) // 2))
        return canvas
    canvas = Image.new("RGB", (CELL, CELL), (60, 60, 60))
    ImageDraw.Draw(canvas).text((10, CELL // 2), "missing", fill=(255, 255, 255), font=font)
    return canvas


tasks = [t for _, names in GROUPS for t in names]
cols = 1 + len(METHODS)  # source + methods
sheet = Image.new("RGB", (LABEL_W + cols * CELL, HEADER_H + len(tasks) * CELL), (255, 255, 255))
draw = ImageDraw.Draw(sheet)
for i, name in enumerate(["source"] + [d for _, d in METHODS]):
    draw.text((LABEL_W + i * CELL + 8, 10), name, fill=(0, 0, 0), font=font)

row = 0
group_of = {t: g for g, names in GROUPS for t in names}
for task in tasks:
    meta = ROOT / "outputs/pretty_matrix" / task / "base_only" / f"seed_{SEED}" / "metadata.json"
    src_path = None
    if meta.is_file():
        src = json.loads(meta.read_text()).get("source_image") or json.loads(meta.read_text()).get("image")
        if src:
            src_path = Path(src)
    y = HEADER_H + row * CELL
    draw.text((6, y + 8), group_of[task], fill=(0, 0, 0), font=font)
    draw.text((6, y + 24), task[:34], fill=(60, 60, 60), font=font)
    sheet.paste(cell_image(src_path) if src_path else cell_image(Path("/nonexistent")), (LABEL_W, y))
    for j, (method, _) in enumerate(METHODS):
        res = ROOT / "outputs/pretty_matrix" / task / method / f"seed_{SEED}" / "result.png"
        sheet.paste(cell_image(res), (LABEL_W + (j + 1) * CELL, y))
    row += 1

sheet_path = OUT_DIR / "e1_t1_t4_seed10_comparison_sheet.png"
sheet.save(sheet_path)
print("sheet:", sheet_path, sheet.size)


def agg(csv_path: Path, method_filter=None):
    """mean of key metrics per (group, method) and overall per method."""
    out = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            m = r["method"]
            if method_filter and m not in method_filter:
                continue
            t = r["task"]
            if t not in group_of:
                continue
            for key in ("outside_mask_l1", "inside_mask_l1", "source_ssim_luma"):
                if r.get(key):
                    out[(group_of[t], m)][key].append(float(r[key]))
                    out[("ALL", m)][key].append(float(r[key]))
    return out


def emit(table, label):
    print(f"\n## {label}")
    print("| Group | Method | n | Outside L1 | Inside L1 | Source SSIM |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for (grp, m), vals in sorted(table.items()):
        n = len(vals["outside_mask_l1"])
        mean = lambda k: sum(vals[k]) / len(vals[k]) if vals[k] else float("nan")
        print(f"| {grp} | {m} | {n} | {mean('outside_mask_l1'):.4f} | {mean('inside_mask_l1'):.4f} | {mean('source_ssim_luma'):.4f} |")


emit(agg(PACK / "e4_t1_t4_reconstruction_floor_metrics.csv"), "base_only reconstruction floor (T1-T4, 3 seeds)")
emit(agg(PACK / "e4_t1_t4_controller_base_metrics.csv"), "support_v3 fixed vs DeCE-RF (T1-T4, 3 seeds)")
emit(agg(PACK / "e2_support_matched_t1_t4_fixed_mask_metrics.csv",
         {"direct_target_raw", "support_v3_controller_rmsgap"}), "direct_target_raw vs DeCE-RF (T1-T4, 3 seeds)")
