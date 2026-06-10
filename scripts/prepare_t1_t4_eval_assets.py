from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path.cwd()
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
MATRIX = ROOT / "outputs" / "pretty_matrix"
CANVAS = 512
SEEDS = ("10", "11", "12")

TASKS = {
    "cat_crown": "data/paper_images/cat_sitting_in_grass.jpg",
    "dog_bow_tie_phase2": "data/paper_images/dog_sitting_cc0.jpg",
    "dog_front_sunglasses_phase2": "data/pretty_free_candidates/commons_dog_front_medusa.jpg",
    "bowl_apple_inside": "data/pretty_free_candidates/pexels_empty_ceramic_bowl_phase1.jpg",
    "white_bowl_orange_tabletop_phase2": "data/phase2_candidates/pexels_sideview_bowl_wood_table_26161017.jpg",
    "brown_bowl_lemon_phase2": "data/phase2_candidates/pexels_wooden_bowls_topdown_6962757.jpg",
    "tshirt_star": "data/pretty_free_candidates/pexels_person_white_tshirt_blue_jeans_8217483.jpg",
    "mug_heart": "data/pretty_free_candidates/pexels_white_mug_6312107.jpg",
    "tote_leaf": "data/pretty_free_candidates/pexels_white_tote_bag_4068314.jpg",
    "red_office_chair_to_blue_office_chair": "data/pretty_free_candidates/unsplash_red_office_chair_concrete_lvVWRzm_NwY.jpg",
    "green_mug_orange_phase2": "data/pretty_free_candidates/pexels_green_mug_marble_7828522.jpg",
    "yellow_vase_blue_phase2": "data/pretty_free_candidates/pexels_yellow_ceramic_vase_8356822.jpg",
}


def fit_to_canvas(image: Image.Image, mode: str) -> Image.Image:
    image = image.convert(mode)
    scale = min(CANVAS / image.width, CANVAS / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resample = Image.Resampling.BICUBIC if mode == "RGB" else Image.Resampling.BILINEAR
    fitted = image.resize(size, resample)
    fill = (245, 245, 245) if mode == "RGB" else 0
    canvas = Image.new(mode, (CANVAS, CANVAS), fill)
    canvas.paste(fitted, ((CANVAS - size[0]) // 2, (CANVAS - size[1]) // 2))
    return canvas


def load_mask(path: Path) -> Image.Image:
    image = Image.open(path).convert("L")
    return fit_to_canvas(image, "L")


def aggregate_eval_mask(task: str) -> tuple[Image.Image, list[str]]:
    paths: list[Path] = []
    for seed in SEEDS:
        path = MATRIX / task / "support_v3_controller_rmsgap" / f"seed_{seed}" / "masks" / "operation_v3_edit_mask.png"
        if path.is_file():
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"no operation_v3_edit_mask masks found for {task}")
    masks = [np.asarray(load_mask(path), dtype=np.float32) / 255.0 for path in paths]
    mean_mask = np.mean(np.stack(masks, axis=0), axis=0)
    # Fixed eval mask proxy for T1-T4 reruns: stable region present across seeds.
    binary = (mean_mask >= 0.35).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L"), [str(path.relative_to(ROOT)) for path in paths]


def main() -> int:
    source_dir = EXP / "normalized_512" / "sources"
    mask_dir = EXP / "normalized_512" / "eval_masks"
    source_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, rel_source in TASKS.items():
        source_path = ROOT / rel_source
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_out = source_dir / f"{task}.png"
        mask_out = mask_dir / f"{task}_eval_mask.png"
        fit_to_canvas(Image.open(source_path), "RGB").save(source_out)
        mask, mask_sources = aggregate_eval_mask(task)
        mask.save(mask_out)
        rows.append(
            {
                "task": task,
                "source_image": rel_source,
                "normalized_source": str(source_out.relative_to(ROOT)),
                "eval_mask": str(mask_out.relative_to(ROOT)),
                "eval_mask_source": "mean support_v3_controller_rmsgap operation_v3_edit_mask over seeds 10/11/12, threshold 0.35",
                "mask_inputs": ";".join(mask_sources),
            }
        )
    manifest = EXP / "normalized_512" / "t1_t4_eval_assets_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "scope": "Phase2 T1-T4 rerun eval assets",
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "canvas": CANVAS,
        "eval_mask_policy": "seed-aggregated DeCE-RF operation support proxy; use for rerun diagnostics only",
        "manifest": str(manifest.relative_to(ROOT)),
    }
    (EXP / "normalized_512" / "t1_t4_eval_assets_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest}")
    print(f"tasks={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
