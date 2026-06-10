from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TASKS = ["cat_crown", "tshirt_star", "backpack_remove_toy_charm"]
SEEDS = ["10", "11"]
METHODS = [
    "direct_target_raw",
    "direct_target_mask_blend",
    "flowedit_mask_blend",
    "support_v3_controller_rmsgap",
]
CANVAS = 512


def fit_to_canvas(path: Path, mode: str = "RGB") -> Image.Image:
    image = Image.open(path).convert(mode)
    scale = min(CANVAS / image.width, CANVAS / image.height)
    fitted_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resample = Image.Resampling.BICUBIC if mode == "RGB" else Image.Resampling.BILINEAR
    fitted = image.resize(fitted_size, resample)
    background = (0, 0, 0) if mode == "RGB" else 0
    canvas = Image.new(mode, (CANVAS, CANVAS), background)
    offset = ((CANVAS - fitted_size[0]) // 2, (CANVAS - fitted_size[1]) // 2)
    canvas.paste(fitted, offset)
    return canvas


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_prompts(run_dir: Path) -> tuple[str, str]:
    meta = load_json(run_dir / "metadata.json")
    return (
        meta.get("source_prompt") or meta.get("effective_source_prompt") or "",
        meta.get("target_prompt") or meta.get("effective_target_prompt") or "",
    )


def source_for_task(exp: Path, task: str) -> Image.Image:
    return Image.open(exp / "normalized_512" / "sources" / f"{task}.png").convert("RGB")


def mask_for_task(exp: Path, task: str) -> Image.Image:
    return Image.open(exp / "normalized_512" / "eval_masks" / f"{task}_eval_mask.png").convert("L")


def blend_with_mask(source: Image.Image, edited: Image.Image, mask: Image.Image) -> Image.Image:
    source_resized = source.convert("RGB").resize(edited.size, Image.Resampling.BICUBIC)
    mask_resized = mask.convert("L").resize(edited.size, Image.Resampling.BILINEAR)
    return Image.composite(edited.convert("RGB"), source_resized, mask_resized)


def make_run(
    *,
    root: Path,
    exp: Path,
    out_root: Path,
    task: str,
    method: str,
    seed: str,
    result: Image.Image,
    source_path: Path,
    source_prompt: str,
    target_prompt: str,
    source_run: Path,
    support_condition: str,
    notes: str,
) -> dict[str, str]:
    run_dir = out_root / task / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result.save(run_dir / "result.png")
    (run_dir / "stats.json").write_text(
        json.dumps({"steps": [], "note": "E2.4 support-matched diagnostic output."}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task": task,
                "method": method,
                "seed": int(seed),
                "image": str(source_path),
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "backbone": "SD3",
                "resolution": [CANVAS, CANVAS],
                "e2_layer": "E2.4 support-matched diagnostic",
                "support_condition": support_condition,
                "normalization_policy": "512x512 normalized source/result/mask canvas; original outputs retained elsewhere.",
                "source_run": str(source_run),
                "notes": notes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(
        f"posthoc_e2_4_diagnostic method={method} task={task} seed={seed} source_run={source_run}\n",
        encoding="utf-8",
    )
    return {
        "task": task,
        "method": method,
        "seed": seed,
        "result": str((run_dir / "result.png").relative_to(root)),
        "metadata": str((run_dir / "metadata.json").relative_to(root)),
        "command": str((run_dir / "command.txt").relative_to(root)),
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
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    return canvas


def make_seed_grid(root: Path, exp: Path, out_root: Path, seed: str) -> Path:
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
        grid.paste(label_cell(source_for_task(exp, task), task), (0, y))
        for col_idx, method in enumerate(METHODS, start=1):
            image = fit_to_canvas(out_root / task / method / f"seed_{seed}" / "result.png")
            grid.paste(label_cell(image, method), (col_idx * CANVAS, y))
    grid_path = exp / "visual_audit" / "e2_support_matched_grids" / f"e2_4_support_matched_seed{seed}_grid.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(grid_path)
    return grid_path


def main() -> int:
    root = Path.cwd()
    exp = root / "experiments" / "support_v3_2026-06-02"
    out_root = root / "outputs" / "e2_support_matched_diagnostic"
    e2_manifest = exp / "e2_strict_rf_baseline_manifest.csv"
    with e2_manifest.open(newline="", encoding="utf-8") as handle:
        e2_rows = list(csv.DictReader(handle))
    source_paths = {}
    for row in e2_rows:
        if row.get("task") in TASKS and row.get("source_image"):
            source_paths.setdefault(row["task"], root / row["source_image"])
    rows = []

    for task in TASKS:
        source_path = source_paths[task]
        source = Image.open(source_path).convert("RGB")
        mask = mask_for_task(exp, task)
        for seed in SEEDS:
            direct_dir = root / "outputs" / "pretty_matrix" / task / "direct_target" / f"seed_{seed}"
            dece_dir = root / "outputs" / "e2_rf_comparison" / task / "support_v3_controller_rmsgap" / f"seed_{seed}"
            flowedit_dir = root / "outputs" / "e2_rf_comparison" / task / "flowedit" / f"seed_{seed}"
            source_prompt, target_prompt = read_prompts(direct_dir)

            direct_img = Image.open(direct_dir / "result.png").convert("RGB")
            flowedit_img = Image.open(flowedit_dir / "result.png").convert("RGB")
            dece_img = Image.open(dece_dir / "result.png").convert("RGB")

            rows.append(
                make_run(
                    root=root,
                    exp=exp,
                    out_root=out_root,
                    task=task,
                    method="direct_target_raw",
                    seed=seed,
                    result=direct_img,
                    source_path=source_path,
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                    source_run=direct_dir,
                    support_condition="none",
                    notes="Raw direct target SD3 row; no support input.",
                )
            )
            rows.append(
                make_run(
                    root=root,
                    exp=exp,
                    out_root=out_root,
                    task=task,
                    method="direct_target_mask_blend",
                    seed=seed,
                    result=blend_with_mask(source, direct_img, mask),
                    source_path=source_path,
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                    source_run=direct_dir,
                    support_condition="posthoc fixed binary edit mask blend",
                    notes="Diagnostic only: output = M_edit * direct_target + (1-M_edit) * source.",
                )
            )
            rows.append(
                make_run(
                    root=root,
                    exp=exp,
                    out_root=out_root,
                    task=task,
                    method="flowedit_mask_blend",
                    seed=seed,
                    result=blend_with_mask(source, flowedit_img, mask),
                    source_path=source_path,
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                    source_run=flowedit_dir,
                    support_condition="posthoc fixed binary edit mask blend",
                    notes="Diagnostic only: output = M_edit * FlowEdit + (1-M_edit) * source.",
                )
            )
            rows.append(
                make_run(
                    root=root,
                    exp=exp,
                    out_root=out_root,
                    task=task,
                    method="support_v3_controller_rmsgap",
                    seed=seed,
                    result=dece_img,
                    source_path=source_path,
                    source_prompt=source_prompt,
                    target_prompt=target_prompt,
                    source_run=dece_dir,
                    support_condition="operation support plus clean-estimate feedback/projection",
                    notes="Full DeCE-RF row.",
                )
            )

    manifest = exp / "e2_support_matched_diagnostic_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for seed in SEEDS:
        make_seed_grid(root, exp, out_root, seed)
    print(f"wrote {manifest}")
    print(f"runs {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
