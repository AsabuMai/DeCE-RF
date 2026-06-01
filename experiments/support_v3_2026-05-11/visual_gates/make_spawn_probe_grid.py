from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

root = Path("/workspace/rf_h_edit")
tasks = [
    ("web_chair_cushion", "chair + cushion"),
    ("web_bowl_spoon", "bowl + spoon"),
    ("web_vase_flowers", "vase + flowers"),
]
methods = [
    ("source", "source"),
    ("support_v3_controller_rmsgap", "old DeCE"),
    ("support_v3_controller_rmsgap_add_editor_v1", "add_v1"),
    ("support_v3_controller_rmsgap_add_editor_v1_spawn", "spawn"),
    ("support_v3_controller_rmsgap_add_editor_v1_spawn_soft", "spawnsoft"),
    ("support_v3_controller_rmsgap_add_editor_v1_hostspawn", "hostspawn"),
    ("support_v3_controller_rmsgap_add_editor_v1_hostwide", "hostwide"),
    ("support_v3_controller_rmsgap_add_editor_v1_spawnlower", "spawnlower"),
    ("support_v3_controller_rmsgap_add_editor_v1_topcontact", "topcontact"),
    ("support_v3_controller_rmsgap_add_editor_v2", "add_v2"),
]
cell_w, cell_h = 280, 260
left_w, label_h = 140, 38
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
            meta = root / "outputs" / "pretty_matrix" / task / "support_v3_controller_rmsgap_add_editor_v1" / "seed_10" / "metadata.json"
            path = Path(json.loads(meta.read_text())["image"])
        else:
            path = root / "outputs" / "pretty_matrix" / task / method / "seed_10" / "result.png"
        if not path.exists():
            draw.text((x + 8, y + 8), "missing", fill=(180, 0, 0), font=font)
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
        tile.paste(img, ((cell_w - img.width) // 2, (cell_h - img.height) // 2))
        canvas.paste(tile, (x, y))
out = root / "experiments" / "support_v3_2026-05-11" / "visual_gates" / "add_spawn_probe_seed10_grid.png"
out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out)
print(out)
