#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


TASKS = {
    "cat_crown": {
        "source_image": "data/paper_images/cat_sitting_in_grass.jpg",
        "source_prompt": "A photo of a cat sitting in grass.",
        "target_prompt": "A photo of the same cat sitting in the same grass, wearing a small golden crown on its head.",
    },
    "dog_sunglasses": {
        "source_image": "data/pretty_free_candidates/unsplash_dog_front_malinois_PGlA5efHOiI.jpg",
        "source_prompt": "A front-facing portrait of a dog in snow.",
        "target_prompt": "A front-facing portrait of the same dog wearing black sunglasses in snow.",
    },
    "mug_heart": {
        "source_image": "data/pretty_free_candidates/pexels_white_mug_6312107.jpg",
        "source_prompt": "A minimalist photo of a plain white ceramic mug on a grey background.",
        "target_prompt": "A minimalist photo of the same white ceramic mug with a small red heart printed on the front, on the same grey background.",
    },
    "backpack_remove_toy_charm": {
        "source_image": "data/pretty_free_candidates/unsplash_backpack_keychain_njwnKDUDKNM.jpg",
        "source_prompt": "A close-up photo of a grey backpack with a yellow dangling toy charm attached to a pink keychain strap.",
        "target_prompt": "A close-up photo of the same grey backpack with the yellow dangling toy charm removed, pink strap, zipper, and fabric preserved.",
    },
}

BASELINES = ["flowedit", "splitflow", "fireflow", "rf_solver_edit", "reflex", "steerflow"]

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize matched external-baseline parity manifest.")
    parser.add_argument("--baselines", default=" ".join(BASELINES))
    parser.add_argument("--tasks", default=" ".join(TASKS))
    parser.add_argument("--seeds", default="10 11 12")
    parser.add_argument("--outputs-dir", default="outputs/baselines")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; use --overwrite to replace it")

    baselines = [item.strip() for item in args.baselines.split() if item.strip()]
    tasks = [item.strip() for item in args.tasks.split() if item.strip()]
    seeds = [item.strip().removeprefix("seed_") for item in args.seeds.split() if item.strip()]

    unknown_tasks = [task for task in tasks if task not in TASKS]
    if unknown_tasks:
        raise SystemExit(f"Unknown tasks: {', '.join(unknown_tasks)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for baseline in baselines:
            for task in tasks:
                task_info = TASKS[task]
                for seed in seeds:
                    run_dir = f"{args.outputs_dir}/{baseline}/{task}/seed_{seed}"
                    writer.writerow(
                        {
                            "baseline": baseline,
                            "task": task,
                            "seed": seed,
                            "status": "pending",
                            "source_image": task_info["source_image"],
                            "source_prompt": task_info["source_prompt"],
                            "target_prompt": task_info["target_prompt"],
                            "result_image": f"{run_dir}/result.png",
                            "metadata": f"{run_dir}/metadata.json",
                            "command": f"{run_dir}/command.txt",
                            "matched_conditions": "",
                            "failure_reason": "",
                            "notes": "",
                        }
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
