from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
OUT = EXP / "e5_boundary_extension_t1_t4"
MATRIX = ROOT / "outputs" / "pretty_matrix"
METHOD = "support_v3_controller_rmsgap"
SEEDS = ["10", "11", "12"]
TASKS = [
    "cat_crown",
    "dog_bow_tie_phase2",
    "dog_front_sunglasses_phase2",
    "bowl_apple_inside",
    "white_bowl_orange_tabletop_phase2",
    "brown_bowl_lemon_phase2",
    "tshirt_star",
    "mug_heart",
    "tote_leaf",
    "red_office_chair_to_blue_office_chair",
    "green_mug_orange_phase2",
    "yellow_vase_blue_phase2",
]

TAXONOMY = {
    "cat_crown": ("positive_core", "above-host accessory", "canonical attached-accessory success case"),
    "dog_bow_tie_phase2": ("extension_core", "below-host accessory", "tests neck/chest accessory geometry"),
    "dog_front_sunglasses_phase2": ("boundary_core", "face-attached accessory", "strict alignment and identity preservation"),
    "bowl_apple_inside": ("positive_core", "inside-container insertion", "canonical container insertion"),
    "white_bowl_orange_tabletop_phase2": ("boundary_core", "surface placement clarity", "orange visibility and scale are human-review sensitive"),
    "brown_bowl_lemon_phase2": ("extension_core", "inside-container insertion", "small bowl interior placement"),
    "tshirt_star": ("boundary_core", "decal naturalness", "hard-edge symbol can look pasted if texture/shading is weak"),
    "mug_heart": ("positive_core", "compact decal", "small hard-edge decal on curved mug"),
    "tote_leaf": ("extension_core", "fabric-panel decal", "larger flat tote panel"),
    "red_office_chair_to_blue_office_chair": ("positive_core", "object recolor", "chair shell recolor with metal/floor preservation"),
    "green_mug_orange_phase2": ("boundary_core", "object recolor", "edge grey and orange clarity are human-review sensitive"),
    "yellow_vase_blue_phase2": ("boundary_core", "object recolor", "fabric boundary and matte blue realism are human-review sensitive"),
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def row_dir(task: str, seed: str) -> Path:
    return MATRIX / task / METHOD / f"seed_{seed}"


def complete(path: Path) -> bool:
    return all((path / name).is_file() for name in ["result.png", "metadata.json", "stats.json", "command.txt"])


def source_image_for(path: Path) -> Path | None:
    meta = read_json(path / "metadata.json")
    for key in ["image", "source_image"]:
        value = meta.get(key)
        if isinstance(value, str) and value:
            candidate = Path(value)
            if candidate.exists():
                return candidate
    return None


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_manifest() -> list[dict[str, str]]:
    rows = []
    for task in TASKS:
        category, label, message = TAXONOMY[task]
        for seed in SEEDS:
            path = row_dir(task, seed)
            rows.append(
                {
                    "task": task,
                    "method": METHOD,
                    "seed": seed,
                    "complete": str(complete(path)),
                    "result": str((path / "result.png").relative_to(ROOT)),
                    "metadata": str((path / "metadata.json").relative_to(ROOT)),
                    "stats": str((path / "stats.json").relative_to(ROOT)),
                    "category": category,
                    "boundary_label": label,
                    "paper_message": message,
                }
            )
    return rows


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, (248, 248, 248))
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_figure() -> Path:
    selected = [
        ("Positive", "cat_crown"),
        ("Insertion boundary", "white_bowl_orange_tabletop_phase2"),
        ("Decal boundary", "tshirt_star"),
        ("Recolor boundary", "yellow_vase_blue_phase2"),
    ]
    seed = "10"
    thumb = (300, 300)
    label_w = 500
    header_h = 60
    width = thumb[0] * 2 + label_w
    height = header_h + thumb[1] * len(selected)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = small = ImageFont.load_default()
    draw.text((12, 18), "Source", fill=(20, 20, 20), font=font)
    draw.text((thumb[0] + 12, 18), "DeCE-RF seed10", fill=(20, 20, 20), font=font)
    draw.text((thumb[0] * 2 + 12, 18), "T1-T4 boundary label", fill=(20, 20, 20), font=font)
    for idx, (title, task) in enumerate(selected):
        y = header_h + idx * thumb[1]
        run = row_dir(task, seed)
        source = source_image_for(run)
        if source:
            canvas.paste(fit_image(source, thumb), (0, y))
        result = run / "result.png"
        if result.exists():
            canvas.paste(fit_image(result, thumb), (thumb[0], y))
        _, label, message = TAXONOMY[task]
        text = f"{title}\n{task}\n{label}\n\n{message}"
        draw.multiline_text((thumb[0] * 2 + 14, y + 24), "\n".join(textwrap.wrap(text, width=48)), fill=(20, 20, 20), font=small, spacing=5)
    path = OUT / "e5_t1_t4_boundary_extension_seed10.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    write_csv(OUT / "e5_t1_t4_selected_manifest.csv", manifest)
    taxonomy_rows = [
        {"task": task, "category": values[0], "boundary_label": values[1], "paper_message": values[2]}
        for task, values in TAXONOMY.items()
    ]
    write_csv(OUT / "e5_t1_t4_failure_taxonomy.csv", taxonomy_rows)
    figure = make_figure()
    complete_rows = sum(1 for row in manifest if row["complete"] == "True")
    summary = OUT / "e5_t1_t4_boundary_extension_summary.md"
    lines = [
        "# E5 T1-T4 Boundary, Extension, And Failure Cases",
        "",
        "Date: 2026-06-09",
        "",
        f"Selected outputs: {complete_rows}/{len(manifest)} complete.",
        f"Figure candidate: `{figure.relative_to(ROOT)}`",
        "",
        "| Task | Seeds | Category | Boundary label |",
        "| --- | ---: | --- | --- |",
    ]
    for task in TASKS:
        count = sum(1 for row in manifest if row["task"] == task and row["complete"] == "True")
        category, label, _ = TAXONOMY[task]
        lines.append(f"| `{task}` | {count}/3 | {category} | {label} |")
    lines.append("")
    lines.append("Claim boundary: this T1-T4 E5 rerun is a boundary/extension audit over current core tasks, not the old T5/T6 removal/completion extension package.")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    complete = OUT / "e5_t1_t4_boundary_extension_complete_2026-06-09.md"
    complete.write_text(
        "\n".join(
            [
                "# E5 T1-T4 Completion Audit",
                "",
                f"Selected outputs complete: {complete_rows}/{len(manifest)}",
                f"Manifest: `{(OUT / 'e5_t1_t4_selected_manifest.csv').relative_to(ROOT)}`",
                f"Taxonomy: `{(OUT / 'e5_t1_t4_failure_taxonomy.csv').relative_to(ROOT)}`",
                f"Summary: `{summary.relative_to(ROOT)}`",
                f"Figure candidate: `{figure.relative_to(ROOT)}`",
                "",
                "Claim boundary: T1-T4 boundary/extension audit only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"complete={complete_rows}/{len(manifest)}")
    print(summary)
    print(complete)
    return 0 if complete_rows == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
