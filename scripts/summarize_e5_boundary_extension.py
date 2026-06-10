from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
OUT = EXP / "e5_boundary_extension"
MATRIX = ROOT / "outputs" / "pretty_matrix"

SEEDS = ["10", "11", "12"]
BASE_METHOD = "support_v3_controller_rmsgap"
GATED_METHOD = "support_v3_controller_rmsgap_completion_clean_delta_gated_highconf"
STAR_METHOD = "support_v3_controller_rmsgap_replace_editor_v1"

BASE_TASKS = [
    "laptop_remove_sticker",
    "fridge_remove_yellow_magnet",
    "fridge_remove_peach_magnet",
    "whiteboard_remove_yellow_letter",
    "dog_remove_tennis_ball",
    "dog_replace_tennis_ball_star",
]
GATED_TASKS = [
    "laptop_remove_sticker",
    "fridge_remove_yellow_magnet",
    "fridge_remove_peach_magnet",
    "whiteboard_remove_yellow_letter",
    "dog_remove_tennis_ball",
]
STAR_TASKS = ["whiteboard_probe_red_star_sticker"]

TASK_LABELS = {
    "laptop_remove_sticker": "laptop sticker removal",
    "fridge_remove_yellow_magnet": "fridge yellow magnet removal",
    "fridge_remove_peach_magnet": "fridge peach magnet removal",
    "whiteboard_remove_yellow_letter": "whiteboard glyph removal",
    "dog_remove_tennis_ball": "dog tennis-ball removal",
    "dog_replace_tennis_ball_star": "dog ball-to-star replacement",
    "whiteboard_probe_red_star_sticker": "whiteboard red-star replacement",
}

TAXONOMY = {
    "laptop_remove_sticker": ("positive_extension", "high-confidence completion prior", "extension can help when completion prior is reliable"),
    "whiteboard_probe_red_star_sticker": ("positive_extension", "replacement target route", "replacement route is useful but separately named"),
    "dog_remove_tennis_ball": ("failure_limit", "removal completion failure", "localization is not enough when the host mouth/fur must be synthesized"),
    "whiteboard_remove_yellow_letter": ("failure_limit", "semantic glyph hallucination", "blank removal in text-like fields is outside the claim"),
    "fridge_remove_yellow_magnet": ("failure_limit", "cluttered-surface damage", "support may be accurate while completion damages cluttered surfaces"),
    "fridge_remove_peach_magnet": ("failure_limit", "cluttered-surface damage", "support may be accurate while completion damages cluttered surfaces"),
    "dog_replace_tennis_ball_star": ("failure_limit", "replacement ambiguity", "target pressure can fire without clean replacement"),
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def row_dir(task: str, method: str, seed: str) -> Path:
    return MATRIX / task / method / f"seed_{seed}"


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
    command = path / "command.txt"
    if command.exists():
        text = command.read_text(encoding="utf-8", errors="ignore")
        parts = text.split()
        if "--image" in parts:
            idx = parts.index("--image")
            if idx + 1 < len(parts):
                candidate = Path(parts[idx + 1])
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
    rows: list[dict[str, str]] = []
    specs = []
    specs += [(task, BASE_METHOD, "base_dece_rf") for task in BASE_TASKS]
    specs += [(task, GATED_METHOD, "completion_prior_route") for task in GATED_TASKS]
    specs += [(task, STAR_METHOD, "replacement_target_route") for task in STAR_TASKS]
    for task, method, route in specs:
        category, failure_type, message = TAXONOMY.get(task, ("", "", ""))
        for seed in SEEDS:
            path = row_dir(task, method, seed)
            rows.append(
                {
                    "task": task,
                    "label": TASK_LABELS.get(task, task),
                    "method": method,
                    "route": route,
                    "seed": seed,
                    "complete": str(complete(path)),
                    "result": str((path / "result.png").relative_to(ROOT)),
                    "metadata": str((path / "metadata.json").relative_to(ROOT)),
                    "stats": str((path / "stats.json").relative_to(ROOT)),
                    "category": category,
                    "failure_type": failure_type,
                    "paper_message": message,
                }
            )
    return rows


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, (248, 248, 248))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_figure6() -> Path:
    rows = [
        ("Positive extension\nreliable completion prior", "laptop_remove_sticker", GATED_METHOD, "high-confidence completion prior\nlaptop sticker removal"),
        ("Limit\nglyph removal", "whiteboard_remove_yellow_letter", BASE_METHOD, "semantic glyph hallucination\nwhiteboard glyph removal"),
        ("Limit\nreplacement ambiguity", "dog_replace_tennis_ball_star", BASE_METHOD, "replacement ambiguity\ndog ball-to-star replacement"),
    ]
    seed = "10"
    thumb = (300, 300)
    label_w = 430
    label_h = 104
    header_h = 64
    cols = ["Source", "E5 output", "Scope label"]
    width = thumb[0] * 2 + label_w
    height = header_h + (thumb[1] + label_h) * len(rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = small = ImageFont.load_default()
    draw.text((12, 20), cols[0], fill=(20, 20, 20), font=font)
    draw.text((thumb[0] + 12, 20), cols[1], fill=(20, 20, 20), font=font)
    draw.text((thumb[0] * 2 + 12, 20), cols[2], fill=(20, 20, 20), font=font)
    for row_idx, (row_title, task, method, scope) in enumerate(rows):
        y = header_h + row_idx * (thumb[1] + label_h)
        out = row_dir(task, method, seed)
        source = source_image_for(out)
        if source:
            canvas.paste(fit_image(source, thumb), (0, y))
        result = out / "result.png"
        if result.exists():
            canvas.paste(fit_image(result, thumb), (thumb[0], y))
        label_x = thumb[0] * 2 + 14
        label_text = f"{row_title}\n\n{scope}"
        draw.multiline_text((label_x, y + 22), label_text, fill=(20, 20, 20), font=small, spacing=6)
        draw.text((12, y + thumb[1] + 10), task, fill=(50, 50, 50), font=small)
        draw.text((thumb[0] + 12, y + thumb[1] + 10), "\n".join(textwrap.wrap(method, width=36)), fill=(50, 50, 50), font=small)
    path = OUT / "e5_figure6_boundary_extension_seed10.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def write_markdown(manifest: list[dict[str, str]], figure: Path) -> Path:
    complete_rows = sum(1 for row in manifest if row["complete"] == "True")
    lines = [
        "# E5 Boundary, Extension, And Failure Cases",
        "",
        "Date: 2026-06-04",
        "",
        f"Selected outputs: {complete_rows}/{len(manifest)} complete.",
        f"Figure 6 candidate: `{figure.relative_to(ROOT)}`",
        "",
        "## Scope",
        "",
        "- Positive extensions are reported separately from the Core-6 main table.",
        "- Failure rows use controlled labels and support limitation wording.",
        "- These rows are not aggregated into the base DeCE-RF mean.",
        "",
        "## Selected Rows",
        "",
        "| Task | Route | Seeds | Category | Failure/extension label |",
        "| --- | --- | ---: | --- | --- |",
    ]
    seen = set()
    for row in manifest:
        key = (row["task"], row["route"])
        if key in seen:
            continue
        seen.add(key)
        count = sum(1 for item in manifest if item["task"] == row["task"] and item["route"] == row["route"] and item["complete"] == "True")
        lines.append(
            f"| `{row['task']}` | {row['route']} | {count}/3 | {row['category']} | {row['failure_type']} |"
        )
    lines.extend(
        [
            "",
            "Paper-safe wording: E5 documents where the method extends or stops. It should be used for Figure 6 and limitations, not as a main-table quantitative claim.",
        ]
    )
    path = OUT / "e5_boundary_extension_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    write_csv(OUT / "e5_selected_manifest.csv", manifest)
    taxonomy_rows = [
        {"task": task, "category": values[0], "failure_type": values[1], "paper_message": values[2]}
        for task, values in TAXONOMY.items()
    ]
    write_csv(OUT / "e5_failure_taxonomy.csv", taxonomy_rows)
    figure = make_figure6()
    summary = write_markdown(manifest, figure)
    complete_rows = sum(1 for row in manifest if row["complete"] == "True")
    completion = OUT / "e5_boundary_extension_complete_2026-06-04.md"
    completion.write_text(
        "\n".join(
            [
                "# E5 Completion Audit",
                "",
                f"Selected outputs complete: {complete_rows}/{len(manifest)}",
                f"Manifest: `{(OUT / 'e5_selected_manifest.csv').relative_to(ROOT)}`",
                f"Taxonomy: `{(OUT / 'e5_failure_taxonomy.csv').relative_to(ROOT)}`",
                f"Summary: `{summary.relative_to(ROOT)}`",
                f"Figure 6 candidate: `{figure.relative_to(ROOT)}`",
                "",
                "Claim boundary: E5 is boundary/extension evidence only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"complete={complete_rows}/{len(manifest)}")
    print(OUT / "e5_selected_manifest.csv")
    print(OUT / "e5_failure_taxonomy.csv")
    print(summary)
    print(figure)
    print(completion)
    return 0 if complete_rows == len(manifest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
