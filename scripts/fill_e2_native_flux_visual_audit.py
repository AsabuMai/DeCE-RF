from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


SCORES = {
    ("fireflow", "cat_crown"): (4, 2, 2, 4, 3, "target_ok_global_identity_drift", "Crown appears, but the cat identity/color and crop drift substantially."),
    ("rf_solver_edit", "cat_crown"): (4, 4, 4, 4, 4, "", "Small crown appears with the best source/layout preservation among native FLUX rows."),
    ("reflex", "cat_crown"): (5, 2, 3, 4, 3, "global_identity_drift", "Crown is clear, but the cat is strongly re-rendered and stylized."),
    ("fireflow", "bowl_apple_inside"): (5, 2, 2, 4, 3, "global_layout_drift", "Apple insertion succeeds, but table layout/crop and surrounding objects change."),
    ("rf_solver_edit", "bowl_apple_inside"): (4, 3, 4, 4, 4, "", "Apple is inserted with comparatively good local behavior, though crop/layout still shifts."),
    ("reflex", "bowl_apple_inside"): (5, 2, 2, 3, 3, "global_layout_drift", "Apple is visible, but the scene is heavily re-rendered and object layout changes."),
    ("fireflow", "tshirt_star"): (5, 4, 4, 4, 5, "", "Star appears cleanly with good preservation; minor background/color drift remains."),
    ("rf_solver_edit", "tshirt_star"): (5, 4, 4, 4, 5, "", "Star appears cleanly with good preservation; minor background/color drift remains."),
    ("reflex", "tshirt_star"): (5, 2, 3, 3, 3, "global_body_pose_drift", "Star appears, but body pose/crop/background are re-rendered."),
    ("fireflow", "red_chair_blue"): (1, 3, 2, 4, 2, "target_miss", "Chair remains largely red; method preserves some scene structure but misses recolor."),
    ("rf_solver_edit", "red_chair_blue"): (1, 3, 2, 3, 2, "target_miss_object_drift", "Chair remains largely red and extra cushion/detail appears."),
    ("reflex", "red_chair_blue"): (4, 2, 3, 4, 3, "global_object_replacement", "Blue chair appears, but the chair geometry and scene are substantially re-rendered."),
    ("fireflow", "pillow_vertical_fabric_strip"): (2, 2, 2, 4, 2, "wrong_edit_global_recolor", "Output turns the pillow blue rather than adding a vertical strip; scene crop changes."),
    ("rf_solver_edit", "pillow_vertical_fabric_strip"): (2, 2, 2, 4, 2, "wrong_edit_global_recolor", "Output turns the pillow blue rather than adding a vertical strip; scene crop changes."),
    ("reflex", "pillow_vertical_fabric_strip"): (3, 1, 1, 3, 2, "global_scene_rewrite", "Blue fabric appears, but the cushion/seat/room are broadly rewritten instead of a local strip."),
    ("fireflow", "backpack_remove_toy_charm"): (1, 4, 2, 4, 2, "target_miss", "Dangling charm is not removed; source structure is mostly preserved."),
    ("rf_solver_edit", "backpack_remove_toy_charm"): (1, 4, 2, 4, 2, "target_miss", "Dangling charm is not removed; source structure is mostly preserved."),
    ("reflex", "backpack_remove_toy_charm"): (1, 2, 2, 3, 1, "target_miss_global_drift", "Charm remains or becomes larger while local backpack details are re-rendered."),
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def fmt(value: float) -> str:
    return f"{value:.2f}"


def main() -> int:
    root = Path.cwd()
    exp = root / "experiments" / "support_v3_2026-06-02"
    template = exp / "visual_audit" / "e2_native_flux_visual_audit_template.csv"
    filled = exp / "visual_audit" / "e2_native_flux_visual_audit_filled.csv"
    summary_csv = exp / "visual_audit" / "e2_native_flux_visual_audit_summary.csv"
    summary_md = exp / "visual_audit" / "e2_native_flux_contextual_conclusion.md"

    rows = list(csv.DictReader(template.open(newline="", encoding="utf-8")))
    for row in rows:
        key = (row["method"], row["task"])
        edit, preserve, locality, artifact, overall, flag, note = SCORES[key]
        row["edit_success_1_5"] = str(edit)
        row["preservation_1_5"] = str(preserve)
        row["locality_1_5"] = str(locality)
        row["artifact_1_5"] = str(artifact)
        row["overall_1_5"] = str(overall)
        row["failure_flag"] = flag
        row["notes"] = note

    with filled.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[("method", row["method"])].append(row)
        grouped[("task", row["task"])].append(row)
        grouped[("method_task", row["method"], row["task"])].append(row)

    summary_rows = []
    for key, items in sorted(grouped.items()):
        kind = key[0]
        label = "/".join(key[1:])
        summary_rows.append(
            {
                "group": kind,
                "label": label,
                "n": str(len(items)),
                "edit_success_mean": fmt(mean([float(r["edit_success_1_5"]) for r in items])),
                "preservation_mean": fmt(mean([float(r["preservation_1_5"]) for r in items])),
                "locality_mean": fmt(mean([float(r["locality_1_5"]) for r in items])),
                "artifact_mean": fmt(mean([float(r["artifact_1_5"]) for r in items])),
                "overall_mean": fmt(mean([float(r["overall_1_5"]) for r in items])),
            }
        )

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    method_rows = [r for r in summary_rows if r["group"] == "method"]
    md_lines = [
        "# E2.3 Native FLUX Contextual Visual Audit",
        "",
        "Scope: FireFlow, RF-Solver-Edit, and ReFlex on strict Core-6, seeds 10/11/12.",
        "Backbone: FLUX.1-dev. Outputs are evaluated as native implementation context, not as E2.2 same-backbone algorithmic evidence.",
        "",
        "## Method Summary",
        "",
        "| Method | n | Edit success | Preservation | Locality | Artifact | Overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(method_rows, key=lambda r: r["label"]):
        md_lines.append(
            f"| {row['label']} | {row['n']} | {row['edit_success_mean']} | {row['preservation_mean']} | "
            f"{row['locality_mean']} | {row['artifact_mean']} | {row['overall_mean']} |"
        )
    md_lines.extend(
        [
            "",
            "## Contextual Conclusion",
            "",
            "The native FLUX rows are runnable and often generate visually polished images, but they do not uniformly solve the localized edit-preserve setting.",
            "RF-Solver-Edit is the strongest contextual row in this audit because it preserves source layout best on localized insertion/decal tasks and passes the cat-crown, bowl-apple, and t-shirt-star probes reasonably well.",
            "FireFlow behaves similarly on the successful insertion/decal cases, but it misses recolor, strip-edit, and removal goals while preserving much of the source.",
            "ReFlex follows target semantics more aggressively on crown, apple, star, and chair-blue cases, but it frequently re-renders object identity, pose, crop, or the broader scene; this makes it a useful native-context baseline rather than a locality-preserving control.",
            "All three native FLUX methods fail the exposed-object removal row visually because the toy charm remains or is re-rendered rather than removed.",
            "",
            "Paper-safe wording: native FLUX editors can produce plausible target-looking images, but under fixed Core-6 evaluation masks their visual audit shows a recurring tradeoff between target formation and preservation/locality. These results should be reported as E2.3 implementation context with explicit backbone and normalization caveats.",
            "",
            f"Filled audit CSV: `{filled}`",
            f"Summary CSV: `{summary_csv}`",
        ]
    )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"wrote {filled}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
