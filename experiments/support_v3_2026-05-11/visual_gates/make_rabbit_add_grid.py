from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

root = Path("/workspace/rf_h_edit")
task = "rabbit_sunglasses"
methods = [
    ("source", "source"),
    ("adaptive_full_generic_support", "generic"),
    ("support_v3_controller_rmsgap", "old DeCE"),
    ("support_v3_controller_rmsgap_add_editor_v1", "add_editor_v1"),
]
cell = 288
label_h = 36
font = ImageFont.load_default()
canvas = Image.new("RGB", (len(methods) * cell, cell + label_h), "white")
draw = ImageDraw.Draw(canvas)
for i, (method, title) in enumerate(methods):
    x = i * cell
    draw.text((x + 8, 12), title, fill=(0, 0, 0), font=font)
    if method == "source":
        meta_path = next((root / "outputs" / "pretty_matrix" / task).glob("*/seed_10/metadata.json"))
        path = Path(json.loads(meta_path.read_text())["image"])
    else:
        path = root / "outputs" / "pretty_matrix" / task / method / "seed_10" / "result.png"
    img = Image.open(path).convert("RGB")
    img.thumbnail((cell, cell), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (cell, cell), (245, 245, 245))
    tile.paste(img, ((cell - img.width) // 2, (cell - img.height) // 2))
    canvas.paste(tile, (x, label_h))
out = root / "experiments" / "support_v3_2026-05-11" / "visual_gates" / "rabbit_sunglasses_add_editor_v1_seed10_grid.png"
out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out)
print(out)
