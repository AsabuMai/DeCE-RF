#!/usr/bin/env python3
"""Initialize the revised strict E2 RF-baseline manifest."""

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
        "target_prompt": "A photo of the same cat sitting in the same grass, wearing a small golden crown on its head.",
    },
    "bowl_apple_inside": {
        "source_image": "data/pretty_free_candidates/pexels_empty_ceramic_bowl_phase1.jpg",
        "source_prompt": "A top-down photo of an empty blue ceramic bowl on a wooden board in a tidy table setting, with no fruit inside the bowl.",
        "target_prompt": "A top-down photo of the same blue ceramic bowl on the same wooden board, with one small red apple centered inside the bowl, while the bowl, board, tableware, leaves, and background remain unchanged.",
    },
    "tshirt_star": {
        "source_image": "data/pretty_free_candidates/pexels_person_white_tshirt_blue_jeans_8217483.jpg",
        "source_prompt": "A close-up fashion photo of a person wearing a plain white t-shirt and blue jeans, with natural fabric folds and soft studio lighting.",
        "target_prompt": "The same person wearing the same white t-shirt and blue jeans, with a clearly visible medium-sized bright red star printed on the center chest, while preserving the fabric folds, shadows, jeans, pose, and background.",
    },
    "red_chair_blue": {
        "source_image": "data/pretty_free_candidates/pexels_red_armchair_room_6758347.jpg",
        "source_prompt": "A photo of a red armless rounded upholstered chair in a stylish room.",
        "target_prompt": "A photo of the same armless rounded upholstered chair in the same stylish room, with only the fabric color changed to deep blue, no armrests added.",
    },
    "pillow_vertical_fabric_strip": {
        "source_image": "data/pretty_free_candidates/pexels_plain_pillow_sofa_phase1.jpg",
        "source_prompt": "A photo of a plain light pillow on a sofa.",
        "target_prompt": "A photo of the same plain light pillow on the same sofa, with a vertical glossy blue silk strip sewn down the center of the pillow from top to bottom, following the pillow perspective and preserving the pillow edges, sofa, shadows, and background.",
    },
    "backpack_remove_toy_charm": {
        "source_image": "data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg",
        "source_prompt": "A close-up photo of a grey backpack with a yellow dangling toy charm attached to a pink keychain strap.",
        "target_prompt": "A close-up photo of the same grey backpack with the yellow dangling toy charm removed, pink strap, zipper, and fabric preserved.",
    },
}


BASELINES = {
    "flowedit": {
        "family": "rf_flow",
        "backbone": "stabilityai/stable-diffusion-3-medium-diffusers",
        "runner": "scripts/archive_legacy_2026-05-11/run_flowedit_baseline.py",
        "control_interface": "source_target_text",
    },
    "splitflow": {
        "family": "rf_flow",
        "backbone": "stabilityai/stable-diffusion-3-medium-diffusers + Mistral prompt decomposition",
        "runner": "scripts/archive_legacy_2026-05-11/run_splitflow_baseline.py",
        "control_interface": "source_target_text",
    },
    "fireflow": {
        "family": "rf_flow",
        "backbone": "black-forest-labs/FLUX.1-dev",
        "runner": "scripts/archive_legacy_2026-05-11/run_fireflow_baseline.py",
        "control_interface": "source_target_text",
    },
    "rf_solver_edit": {
        "family": "rf_solver",
        "backbone": "black-forest-labs/FLUX.1-dev",
        "runner": "scripts/archive_legacy_2026-05-11/run_rf_solver_edit_baseline.py",
        "control_interface": "source_target_text",
    },
    "reflex": {
        "family": "rf_rectified_flow",
        "backbone": "black-forest-labs/FLUX.1-dev",
        "runner": "scripts/archive_legacy_2026-05-11/run_reflex_baseline.py",
        "control_interface": "source_target_text_optional_mask",
    },
    "flowalign": {
        "family": "rf_flow",
        "backbone": "FLUX/SD3 as configured by upstream",
        "runner": "adapter pending",
        "control_interface": "source_target_text",
    },
    "stable_flow": {
        "family": "rf_flow",
        "backbone": "black-forest-labs/FLUX.1-dev",
        "runner": "adapter pending",
        "control_interface": "source_target_text",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv"))
    parser.add_argument("--baselines", default="flowedit splitflow fireflow rf_solver_edit reflex")
    parser.add_argument("--tasks", default="cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_vertical_fabric_strip backpack_remove_toy_charm")
    parser.add_argument("--seeds", default="10 11 12")
    args = parser.parse_args()

    baselines = [item for item in args.baselines.split() if item]
    tasks = [item for item in args.tasks.split() if item]
    seeds = [item.removeprefix("seed_") for item in args.seeds.split() if item]

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
                            "notes": "pending strict generation validation; "
                            + base["family"]
                            + "; "
                            + base["backbone"]
                            + "; "
                            + base["runner"],
                        }
                    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
