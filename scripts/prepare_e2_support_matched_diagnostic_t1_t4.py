from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path.cwd()
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
OUT = ROOT / "outputs" / "e2_support_matched_diagnostic_t1_t4"
CANVAS = 512
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
METHODS = [
    "direct_target_raw",
    "direct_target_mask_blend",
    "flowedit_mask_blend",
    "support_v3_controller_rmsgap",
]


def fit_to_canvas(path: Path, mode: str = "RGB") -> Image.Image:
    image = Image.open(path).convert(mode)
    scale = min(CANVAS / image.width, CANVAS / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resample = Image.Resampling.BICUBIC if mode == "RGB" else Image.Resampling.BILINEAR
    fitted = image.resize(size, resample)
    fill = (245, 245, 245) if mode == "RGB" else 0
    canvas = Image.new(mode, (CANVAS, CANVAS), fill)
    canvas.paste(fitted, ((CANVAS - size[0]) // 2, (CANVAS - size[1]) // 2))
    return canvas


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_prompts(run_dir: Path) -> tuple[str, str]:
    meta = read_json(run_dir / "metadata.json")
    return (
        meta.get("source_prompt") or meta.get("effective_source_prompt") or "",
        meta.get("target_prompt") or meta.get("effective_target_prompt") or "",
    )


def source_for_task(task: str) -> Image.Image:
    return Image.open(EXP / "normalized_512" / "sources" / f"{task}.png").convert("RGB")


def mask_for_task(task: str) -> Image.Image:
    return Image.open(EXP / "normalized_512" / "eval_masks" / f"{task}_eval_mask.png").convert("L")


def blend_with_mask(source: Image.Image, edited: Image.Image, mask: Image.Image) -> Image.Image:
    source_resized = source.convert("RGB").resize(edited.size, Image.Resampling.BICUBIC)
    mask_resized = mask.convert("L").resize(edited.size, Image.Resampling.BILINEAR)
    return Image.composite(edited.convert("RGB"), source_resized, mask_resized)


def make_run(
    *,
    task: str,
    method: str,
    seed: str,
    result: Image.Image,
    source_prompt: str,
    target_prompt: str,
    source_run: Path,
    support_condition: str,
    notes: str,
) -> dict[str, str]:
    run_dir = OUT / task / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result.save(run_dir / "result.png")
    (run_dir / "stats.json").write_text(
        json.dumps({"steps": [], "note": "E2.4 T1-T4 support-matched diagnostic output."}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task": task,
                "method": method,
                "seed": int(seed),
                "image": str(EXP / "normalized_512" / "sources" / f"{task}.png"),
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "backbone": "SD3",
                "resolution": [CANVAS, CANVAS],
                "e2_layer": "E2.4 T1-T4 support-matched diagnostic",
                "support_condition": support_condition,
                "source_run": str(source_run),
                "notes": notes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(
        f"posthoc_e2_4_t1_t4 method={method} task={task} seed={seed} source_run={source_run}\n",
        encoding="utf-8",
    )
    return {
        "task": task,
        "method": method,
        "seed": seed,
        "result": str((run_dir / "result.png").relative_to(ROOT)),
        "metadata": str((run_dir / "metadata.json").relative_to(ROOT)),
        "command": str((run_dir / "command.txt").relative_to(ROOT)),
        "support_condition": support_condition,
        "notes": notes,
    }


def label_cell(image: Image.Image, text: str) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, CANVAS, 34), fill=(0, 0, 0))
    draw.text((8, 8), text[:70], fill=(255, 255, 255), font=font)
    return canvas


def make_seed_grid(seed: str) -> Path:
    cols = ["source", *METHODS]
    header_h = 44
    grid = Image.new("RGB", (CANVAS * len(cols), header_h + CANVAS * len(TASKS)), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
    except Exception:
        font = ImageFont.load_default()
    for idx, col in enumerate(cols):
        draw.rectangle((idx * CANVAS, 0, (idx + 1) * CANVAS, header_h), fill=(32, 32, 32))
        draw.text((idx * CANVAS + 8, 10), col, fill=(255, 255, 255), font=font)
    for row_idx, task in enumerate(TASKS):
        y = header_h + row_idx * CANVAS
        grid.paste(label_cell(source_for_task(task), task), (0, y))
        for col_idx, method in enumerate(METHODS, start=1):
            image = fit_to_canvas(OUT / task / method / f"seed_{seed}" / "result.png")
            grid.paste(label_cell(image, method), (col_idx * CANVAS, y))
    grid_path = EXP / "visual_audit" / "e2_support_matched_t1_t4_grids" / f"e2_4_t1_t4_seed{seed}_grid.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(grid_path)
    return grid_path


def main() -> int:
    rows = []
    for task in TASKS:
        source = source_for_task(task)
        mask = mask_for_task(task)
        for seed in SEEDS:
            direct_dir = ROOT / "outputs" / "pretty_matrix" / task / "direct_target" / f"seed_{seed}"
            dece_dir = ROOT / "outputs" / "pretty_matrix" / task / "support_v3_controller_rmsgap" / f"seed_{seed}"
            flowedit_dir = ROOT / "outputs" / "baselines" / "flowedit" / task / f"seed_{seed}"
            source_prompt, target_prompt = read_prompts(direct_dir)

            direct_img = Image.open(direct_dir / "result.png").convert("RGB")
            flowedit_img = Image.open(flowedit_dir / "result.png").convert("RGB")
            dece_img = Image.open(dece_dir / "result.png").convert("RGB")
            rows.append(
                make_run(
                    task=task,
                    method="direct_target_raw",
                    seed=seed,
                    result=direct_img,
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                    source_run=direct_dir,
                    support_condition="none",
                    notes="Raw direct target SD3 row; no support input.",
                )
            )
            rows.append(
                make_run(
                    task=task,
                    method="direct_target_mask_blend",
                    seed=seed,
                    result=blend_with_mask(source, direct_img, mask),
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                    source_run=direct_dir,
                    support_condition="posthoc fixed binary edit mask blend",
                    notes="Diagnostic only: M_edit * direct_target + (1-M_edit) * source.",
                )
            )
            rows.append(
                make_run(
                    task=task,
                    method="flowedit_mask_blend",
                    seed=seed,
                    result=blend_with_mask(source, flowedit_img, mask),
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                    source_run=flowedit_dir,
                    support_condition="posthoc fixed binary edit mask blend",
                    notes="Diagnostic only: M_edit * FlowEdit + (1-M_edit) * source.",
                )
            )
            rows.append(
                make_run(
                    task=task,
                    method="support_v3_controller_rmsgap",
                    seed=seed,
                    result=dece_img,
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                    source_run=dece_dir,
                    support_condition="operation support plus clean-estimate feedback/projection",
                    notes="Full DeCE-RF row.",
                )
            )

    manifest = EXP / "e2_support_matched_t1_t4_diagnostic_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for seed in SEEDS:
        make_seed_grid(seed)
    print(f"wrote {manifest}")
    print(f"runs={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
