from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

root = Path("/workspace/rf_h_edit")
tasks = [
    ("web_plate_apple", "bowl + apple"),
    ("web_vase_flowers", "vase + flowers"),
    ("web_chair_cushion", "chair + cushion"),
    ("web_frame_landscape", "frame + landscape"),
    ("web_desk_mug", "desk + mug"),
    ("web_wall_clock", "wall + clock"),
    ("web_shelf_books", "shelf + books"),
    ("web_bowl_spoon", "bowl + spoon"),
    ("web_notebook_pen", "notebook + pen"),
]
methods = [
    ("source", "source"),
    ("support_v3_controller_rmsgap", "old DeCE"),
    ("support_v3_controller_rmsgap_add_editor_v2", "add_v2"),
]
cell_w, cell_h = 280, 220
left_w, label_h = 150, 38
font = ImageFont.load_default()
canvas = Image.new("RGB", (left_w + len(methods) * cell_w, label_h + len(tasks) * cell_h), "white")
draw = ImageDraw.Draw(canvas)

for c, (_, title) in enumerate(methods):
    draw.text((left_w + c * cell_w + 8, 14), title, fill=(0, 0, 0), font=font)

for r, (task, label) in enumerate(tasks):
    y = label_h + r * cell_h
    draw.text((8, y + 8), label, fill=(0, 0, 0), font=font)
    for c, (method, _) in enumerate(methods):
        x = left_w + c * cell_w
        if method == "source":
            meta = root / "outputs" / "pretty_matrix" / task / "support_v3_controller_rmsgap_add_editor_v2" / "seed_10" / "metadata.json"
            path = Path(json.loads(meta.read_text())["image"]) if meta.exists() else None
        else:
            path = root / "outputs" / "pretty_matrix" / task / method / "seed_10" / "result.png"
        if path is None or not path.exists():
            draw.text((x + 8, y + 8), "missing", fill=(180, 0, 0), font=font)
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
        tile.paste(img, ((cell_w - img.width) // 2, (cell_h - img.height) // 2))
        canvas.paste(tile, (x, y))

out = root / "experiments" / "support_v3_2026-05-11" / "visual_gates" / "add_v2_full_seed10_grid.png"
out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out)
print(out)
