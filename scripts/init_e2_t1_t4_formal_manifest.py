#!/usr/bin/env python3
"""Initialize the Phase-2 E2 formal T1-T4 external-baseline manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "baseline",
    "task",
    "seed",
    "status",
    "source_image",
    "source_prompt",
    "target_prompt",
    "result_image",
    "metadata",
    "command",
    "matched_conditions",
    "failure_reason",
    "notes",
]


TASKS = {
    "cat_crown": {
        "source_image": "data/paper_images/cat_sitting_in_grass.jpg",
        "source_prompt": "A photo of a cat sitting in grass.",
        "target_prompt": "A photo of the same cat sitting in the same grass, wearing a small golden crown centered on top of its head between the ears.",
    },
    "dog_bow_tie_phase2": {
        "source_image": "data/paper_images/dog_sitting_cc0.jpg",
        "source_prompt": "A photo of a shaggy grey and white dog sitting in grass.",
        "target_prompt": "A photo of the same shaggy grey and white dog sitting in the same grass, wearing a small red bow tie attached at the front of its neck, while the dog, grass, pose, and background remain unchanged.",
    },
    "dog_front_sunglasses_phase2": {
        "source_image": "data/pretty_free_candidates/commons_dog_front_medusa.jpg",
        "source_prompt": "A close-up front-facing portrait of a dog indoors.",
        "target_prompt": "A close-up front-facing portrait of the same dog wearing black sunglasses aligned across both eyes indoors, while the dog face, ears, fur, nose, floor, and lighting remain unchanged.",
    },
    "bowl_apple_inside": {
        "source_image": "data/pretty_free_candidates/pexels_empty_ceramic_bowl_phase1.jpg",
        "source_prompt": "A top-down photo of an empty blue ceramic bowl on a wooden board in a tidy table setting, with no fruit inside the bowl.",
        "target_prompt": "A top-down photo of the same blue ceramic bowl on the same wooden board, with one small red apple centered inside the bowl, while the bowl, board, tableware, leaves, and background remain unchanged.",
    },
    "white_bowl_orange_tabletop_phase2": {
        "source_image": "data/phase2_candidates/pexels_sideview_bowl_wood_table_26161017.jpg",
        "source_prompt": "A side-view photo of a white bowl sitting on a wooden tabletop, with empty wooden table space to the left of the bowl.",
        "target_prompt": "A side-view photo of the same white bowl sitting on the same wooden tabletop, with one medium bright orange fruit resting on the empty wooden table space to the left of the bowl, while the bowl, tabletop, wall, lighting, and background remain unchanged.",
    },
    "brown_bowl_lemon_phase2": {
        "source_image": "data/phase2_candidates/pexels_wooden_bowls_topdown_6962757.jpg",
        "source_prompt": "A top-down photo of a small empty wooden bowl on a pale wooden table beside a larger wooden bowl and folded cloth.",
        "target_prompt": "A top-down photo of the same small wooden bowl on the same pale wooden table, with one small yellow lemon wedge centered inside the small bowl, while the bowls, table, folded cloth, lighting, and background remain unchanged.",
    },
    "tshirt_star": {
        "source_image": "data/pretty_free_candidates/pexels_person_white_tshirt_blue_jeans_8217483.jpg",
        "source_prompt": "A close-up fashion photo of a person wearing a plain white t-shirt and blue jeans, with natural fabric folds and soft studio lighting.",
        "target_prompt": "The same person wearing the same white t-shirt and blue jeans, with a clearly visible medium-sized bright red star printed on the center chest, while preserving the fabric folds, shadows, jeans, pose, and background.",
    },
    "mug_heart": {
        "source_image": "data/pretty_free_candidates/pexels_white_mug_6312107.jpg",
        "source_prompt": "A minimalist photo of a plain white ceramic mug on a grey background.",
        "target_prompt": "A minimalist photo of the same white ceramic mug with a small flat hard-edged solid red heart decal printed cleanly on the front, with sharp printed edges and no glow or blur, while the mug shape, handle, highlights, bottom shadow, grey background, and lighting remain unchanged.",
    },
    "tote_leaf": {
        "source_image": "data/pretty_free_candidates/pexels_white_tote_bag_4068314.jpg",
        "source_prompt": "A photo of a plain beige canvas tote bag held in front of a dark green wall.",
        "target_prompt": "A photo of the same beige canvas tote bag with a large centered dark green leaf logo printed clearly on the front panel, while the hand, straps, bag shape, wall, and lighting remain unchanged.",
    },
    "red_office_chair_to_blue_office_chair": {
        "source_image": "data/pretty_free_candidates/unsplash_red_office_chair_concrete_lvVWRzm_NwY.jpg",
        "source_prompt": "A photo of a red office chair on a concrete floor.",
        "target_prompt": "A photo of the same office chair on the same concrete floor, with only the red plastic seat and back shell changed to deep blue while the silver metal base, wheels, wall, and floor remain unchanged.",
    },
    "green_mug_orange_phase2": {
        "source_image": "data/pretty_free_candidates/pexels_green_mug_marble_7828522.jpg",
        "source_prompt": "A studio photo of a plain green ceramic mug on a white marble block against a pink background.",
        "target_prompt": "A studio photo of the same plain ceramic mug on the same white marble block against the same pink background, with only the mug color changed from green to orange.",
    },
    "yellow_vase_blue_phase2": {
        "source_image": "data/pretty_free_candidates/pexels_yellow_ceramic_vase_8356822.jpg",
        "source_prompt": "A still-life photo of a small yellow ceramic vase resting on white fabric.",
        "target_prompt": "A still-life photo of the same small ceramic vase resting on the same white fabric, with only the vase color changed from yellow to deep blue while the fabric, lighting, and shadows remain unchanged.",
    },
}


BASELINES = {
    "flowedit": {
        "paper_bucket": "E2.2 same-backbone SD3 target-mode RF baseline",
        "backbone": "stabilityai/stable-diffusion-3-medium-diffusers",
        "runner": "scripts/archive_legacy_2026-05-11/run_flowedit_baseline.py",
    },
    "flowalign": {
        "paper_bucket": "E2.2 same-backbone SD3 target-mode RF baseline",
        "backbone": "stabilityai/stable-diffusion-3-medium-diffusers",
        "runner": "scripts/run_flowalign_baseline.py",
    },
    "splitflow": {
        "paper_bucket": "E2.2 same-backbone SD3 target-mode RF baseline",
        "backbone": "stabilityai/stable-diffusion-3-medium-diffusers + prompt decomposition",
        "runner": "scripts/archive_legacy_2026-05-11/run_splitflow_baseline.py",
    },
    "fireflow": {
        "paper_bucket": "E2.3 native-FLUX contextual baseline",
        "backbone": "black-forest-labs/FLUX.1-dev",
        "runner": "scripts/archive_legacy_2026-05-11/run_fireflow_baseline.py",
    },
    "rf_solver_edit": {
        "paper_bucket": "E2.3 native-FLUX contextual baseline",
        "backbone": "black-forest-labs/FLUX.1-dev",
        "runner": "scripts/archive_legacy_2026-05-11/run_rf_solver_edit_baseline.py",
    },
    "reflex": {
        "paper_bucket": "E2.3 native-FLUX contextual baseline",
        "backbone": "black-forest-labs/FLUX.1-dev",
        "runner": "scripts/archive_legacy_2026-05-11/run_reflex_baseline.py",
    },
}


DEFAULT_TASKS = " ".join(TASKS)
DEFAULT_BASELINES = "flowedit flowalign splitflow fireflow rf_solver_edit reflex"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/support_v3_2026-06-02/e2_t1_t4_formal_baseline_manifest.csv"))
    parser.add_argument("--baselines", default=DEFAULT_BASELINES)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--seeds", default="10 11 12")
    args = parser.parse_args()

    baselines = [item for item in args.baselines.split() if item]
    tasks = [item for item in args.tasks.split() if item]
    seeds = [item.removeprefix("seed_") for item in args.seeds.split() if item]

    missing_baselines = sorted(set(baselines) - set(BASELINES))
    missing_tasks = sorted(set(tasks) - set(TASKS))
    if missing_baselines or missing_tasks:
        raise SystemExit(f"Unknown baselines={missing_baselines} tasks={missing_tasks}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for baseline in baselines:
            base = BASELINES[baseline]
            for task in tasks:
                spec = TASKS[task]
                for seed in seeds:
                    writer.writerow(
                        {
                            "baseline": baseline,
                            "task": task,
                            "seed": seed,
                            "status": "pending",
                            "source_image": spec["source_image"],
                            "source_prompt": spec["source_prompt"],
                            "target_prompt": spec["target_prompt"],
                            "result_image": "",
                            "metadata": "",
                            "command": "",
                            "matched_conditions": "",
                            "failure_reason": "",
                            "notes": "; ".join([base["paper_bucket"], base["backbone"], base["runner"]]),
                        }
                    )
    print(f"Wrote {args.out}")
    print(f"rows={len(baselines) * len(tasks) * len(seeds)} baselines={','.join(baselines)} tasks={len(tasks)} seeds={','.join(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
