#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


TASK_ALIASES = {
    "P1": "cat_crown",
    "P2": "dog_sunglasses",
    "P3": "mug_heart",
    "P5": "tshirt_star",
    "P6": "tote_leaf",
    "P7": "backpack_remove_toy_charm",
    "P12": "rabbit_sunglasses",
    "P13": "dog_crown",
}

METHOD_ALIASES = {
    "M0": "base_only",
    "M1": "direct_target",
    "M4": "full",
    "M5": "full_no_ref",
    "M6": "full_no_rec",
    "M7": "full_no_traj",
    "M10": "adaptive_full_generic_support",
    "M16": "adaptive_full_support_v3",
    "M17": "support_v3_fixed",
    "M18": "support_v3_controller_rmsgap",
    "M19": "support_v3_controller_progress",
    "M20": "support_v3_controller_hybrid",
}

FIELDS = [
    "task",
    "method",
    "seed",
    "edit_success_1_5",
    "source_preservation_1_5",
    "locality_1_5",
    "artifact_severity_1_5",
    "placement_quality_1_5",
    "old_object_absence_1_5",
    "new_object_presence_1_5",
    "fill_quality_1_5",
    "overall_1_5",
    "success",
    "failure_flags",
    "review_image",
    "result_image",
    "notes",
]

DEFAULT_FAILURE_FLAGS = (
    "success",
    "semantic_miss",
    "wrong_placement",
    "partial_edit",
    "over_edit",
    "background_drift",
    "identity_drift",
    "artifact",
    "support_failure",
    "removal_failure",
    "old_object_remains",
    "new_object_missing",
)


def parse_items(value: str, aliases: dict[str, str]) -> list[str]:
    items = []
    for raw_item in value.split():
        item = raw_item.strip()
        if not item:
            continue
        items.append(aliases.get(item, item))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize a pretty-matrix visual audit CSV.")
    parser.add_argument("--tasks", default="P1 P2 P3 P7")
    parser.add_argument("--methods", default="M0 M1 M10 M17 M18")
    parser.add_argument("--seeds", default="10 11 12")
    parser.add_argument("--outputs-dir", default="outputs/pretty_matrix")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} exists; use --overwrite to replace it")

    tasks = parse_items(args.tasks, TASK_ALIASES)
    methods = parse_items(args.methods, METHOD_ALIASES)
    seeds = [item.strip().removeprefix("seed_") for item in args.seeds.split() if item.strip()]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for task in tasks:
            review_image = f"{args.outputs_dir}/{task}_seed_method_review.png"
            for method in methods:
                for seed in seeds:
                    writer.writerow(
                        {
                            "task": task,
                            "method": method,
                            "seed": seed,
                            "success": "",
                            "failure_flags": "",
                            "review_image": review_image,
                            "result_image": f"{args.outputs_dir}/{task}/{method}/seed_{seed}/result.png",
                            "notes": f"Allowed failure_flags: {';'.join(DEFAULT_FAILURE_FLAGS)}",
                        }
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
