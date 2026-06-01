from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

root = Path("/workspace/rf_h_edit")
tasks = [
    ("web_chair_cushion", "chair + cushion"),
    ("web_bowl_spoon", "bowl + spoon"),
    ("web_vase_flowers", "vase + flowers"),
    ("web_notebook_pen", "notebook + pen"),
]
methods = [
    ("source", "source"),
    ("adaptive_full_generic_support", "generic"),
    ("support_v3_controller_rmsgap", "old DeCE"),
    ("support_v3_controller_rmsgap_add_editor_v1", "add_editor_v1"),
    ("support_v3_controller_rmsgap_add_editor_v1_compact", "compact"),
    ("support_v3_controller_rmsgap_add_editor_v1_surface", "surface"),
]

cell_w, cell_h = 256, 256
label_h = 42
left_w = 170
pad = 12
font = ImageFont.load_default()

canvas = Image.new(
    "RGB",
    (left_w + len(methods) * cell_w, label_h + len(tasks) * (cell_h + pad)),
    "white",
)
draw = ImageDraw.Draw(canvas)

for c, (_, title) in enumerate(methods):
    x = left_w + c * cell_w
    draw.text((x + 8, 14), title, fill=(0, 0, 0), font=font)

for r, (task, label) in enumerate(tasks):
    y = label_h + r * (cell_h + pad)
    draw.text((8, y + 8), label, fill=(0, 0, 0), font=font)
    for c, (method, _) in enumerate(methods):
        x = left_w + c * cell_w
        if method == "source":
            meta_candidates = list((root / "outputs" / "pretty_matrix" / task).glob("*/seed_10/metadata.json"))
            path = None
            if meta_candidates:
                meta = json.loads(meta_candidates[0].read_text())
                image_path = meta.get("image")
                path = Path(image_path) if image_path else None
        else:
            path = root / "outputs" / "pretty_matrix" / task / method / "seed_10" / "result.png"
        if path is None or not path.exists():
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(190, 190, 190))
            draw.text((x + 8, y + 8), "missing", fill=(180, 0, 0), font=font)
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
        ox = (cell_w - img.width) // 2
        oy = (cell_h - img.height) // 2
        tile.paste(img, (ox, oy))
        canvas.paste(tile, (x, y))

out = root / "experiments" / "support_v3_2026-05-11" / "visual_gates" / "add_editor_v1_fixed_probe_seed10_grid.png"
out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out)
print(out)
