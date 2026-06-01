from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

root = Path("/workspace/rf_h_edit")
method = "support_v3_controller_rmsgap_add_editor_v1"
tasks = [
    ("rabbit_sunglasses", "rabbit + sunglasses"),
    ("web_chair_cushion", "chair + cushion"),
    ("web_bowl_spoon", "bowl + spoon"),
    ("web_vase_flowers", "vase + flowers"),
]
cols = [("source", "source"), ("seed_10", "seed 10"), ("seed_11", "seed 11"), ("seed_12", "seed 12")]

cell_w, cell_h = 260, 220
label_h = 38
left_w = 170
font = ImageFont.load_default()
canvas = Image.new("RGB", (left_w + len(cols) * cell_w, label_h + len(tasks) * cell_h), "white")
draw = ImageDraw.Draw(canvas)

for c, (_, title) in enumerate(cols):
    draw.text((left_w + c * cell_w + 8, 14), title, fill=(0, 0, 0), font=font)

for r, (task, label) in enumerate(tasks):
    y = label_h + r * cell_h
    draw.text((8, y + 8), label, fill=(0, 0, 0), font=font)
    for c, (col, _) in enumerate(cols):
        x = left_w + c * cell_w
        if col == "source":
            meta_path = root / "outputs" / "pretty_matrix" / task / method / "seed_10" / "metadata.json"
            path = Path(json.loads(meta_path.read_text())["image"])
        else:
            path = root / "outputs" / "pretty_matrix" / task / method / col / "result.png"
        if not path.exists():
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(190, 190, 190))
            draw.text((x + 8, y + 8), "missing", fill=(180, 0, 0), font=font)
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
        tile.paste(img, ((cell_w - img.width) // 2, (cell_h - img.height) // 2))
        canvas.paste(tile, (x, y))

out = root / "experiments" / "support_v3_2026-05-11" / "visual_gates" / "add_editor_v1_positive_multiseed_grid.png"
out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out)
print(out)
