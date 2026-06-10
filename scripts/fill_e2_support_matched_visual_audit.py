from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = [
    "direct_target_raw",
    "direct_target_mask_blend",
    "flowedit_mask_blend",
    "support_v3_controller_rmsgap",
]

TASK_ORDER = [
    "cat_crown",
    "tshirt_star",
    "backpack_remove_toy_charm",
]

SCORE_FIELDS = [
    "edit_success",
    "preservation",
    "locality",
    "artifact",
    "overall",
]

METRIC_FIELDS = [
    "outside_mask_l1_mean",
    "inside_mask_l1_mean",
    "source_ssim_luma_mean",
    "outside_mask_l1_excess_preserve_error_mean",
]

SCORES = {
    ("cat_crown", "direct_target_raw"): (
        1,
        1,
        1,
        3,
        1,
        "target_miss_global_identity_drift",
        "Direct target changes cat identity/crop and does not reliably add a crown.",
    ),
    ("cat_crown", "direct_target_mask_blend"): (
        1,
        4,
        3,
        2,
        2,
        "target_miss_blend_boundary",
        "Mask blend preserves the outside region but pastes a mismatched cat-face patch; no crown.",
    ),
    ("cat_crown", "flowedit_mask_blend"): (
        1,
        4,
        3,
        2,
        2,
        "target_miss_blend_boundary",
        "Mask blend preserves the outside region but inserts a mismatched face/eye patch; no crown.",
    ),
    ("cat_crown", "support_v3_fixed"): (
        5,
        4,
        4,
        4,
        4,
        "success",
        "Crown appears on the original cat with good preservation.",
    ),
    ("cat_crown", "support_v3_controller_rmsgap"): (
        5,
        4,
        4,
        4,
        4,
        "success",
        "Crown appears with good preservation and locality.",
    ),
    ("tshirt_star", "direct_target_raw"): (
        1,
        2,
        1,
        4,
        1,
        "target_miss_global_drift",
        "Direct target changes pose/crop and leaves only a tiny mark rather than a clear red star.",
    ),
    ("tshirt_star", "direct_target_mask_blend"): (
        1,
        4,
        3,
        3,
        2,
        "target_miss",
        "Outside is preserved by construction, but the red star is absent or too weak.",
    ),
    ("tshirt_star", "flowedit_mask_blend"): (
        2,
        4,
        3,
        2,
        2,
        "partial_decal_blend_artifact",
        "A red fragment appears in the shirt region, but it is not a coherent star decal.",
    ),
    ("tshirt_star", "support_v3_fixed"): (
        4,
        3,
        4,
        4,
        4,
        "success",
        "A clear red star is added on the shirt with acceptable fabric and pose preservation.",
    ),
    ("tshirt_star", "support_v3_controller_rmsgap"): (
        4,
        3,
        4,
        3,
        4,
        "success",
        "A clear red star is added on the shirt; minor shading/overlay artifacts remain.",
    ),
    ("red_chair_blue", "direct_target_raw"): (
        1,
        1,
        1,
        3,
        1,
        "target_miss_global_drift",
        "Scene is re-rendered and the chair remains mostly red.",
    ),
    ("red_chair_blue", "direct_target_mask_blend"): (
        1,
        4,
        3,
        3,
        2,
        "target_miss",
        "Outside is preserved by construction, but the chair remains red or changes only slightly.",
    ),
    ("red_chair_blue", "flowedit_mask_blend"): (
        2,
        3,
        3,
        2,
        2,
        "partial_wrong_recolor_hard_boundary",
        "Blue patch/stripe appears, but not a coherent full local chair recolor.",
    ),
    ("red_chair_blue", "support_v3_fixed"): (
        4,
        3,
        4,
        3,
        4,
        "success_moderate_preservation",
        "Chair recolor succeeds with moderate scene preservation.",
    ),
    ("red_chair_blue", "support_v3_controller_rmsgap"): (
        4,
        3,
        4,
        3,
        4,
        "success_moderate_preservation",
        "Chair recolor succeeds with moderate scene preservation.",
    ),
    ("backpack_remove_toy_charm", "direct_target_raw"): (
        3,
        2,
        2,
        3,
        2,
        "global_drift_partial_removal",
        "Toy/charm is partly removed or changed, but the backpack and scene drift.",
    ),
    ("backpack_remove_toy_charm", "direct_target_mask_blend"): (
        2,
        4,
        3,
        2,
        2,
        "partial_removal_blend_artifact",
        "Outside is preserved, but the toy/charm remains partially visible or cut at the boundary.",
    ),
    ("backpack_remove_toy_charm", "flowedit_mask_blend"): (
        1,
        3,
        2,
        2,
        1,
        "target_miss_blend_artifact",
        "Blended FlowEdit leaves or worsens the toy/charm region.",
    ),
    ("backpack_remove_toy_charm", "support_v3_fixed"): (
        5,
        4,
        4,
        4,
        5,
        "success",
        "Toy/charm is removed while keeping the backpack and nearby structure mostly intact.",
    ),
    ("backpack_remove_toy_charm", "support_v3_controller_rmsgap"): (
        5,
        4,
        4,
        4,
        5,
        "success",
        "Toy/charm is removed while preserving the backpack and nearby structure.",
    ),
}


def avg(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return sum(values) / len(values) if values else 0.0


def fmt(value: float) -> str:
    return f"{value:.2f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_audit_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for source in manifest_rows:
        key = (source["task"], source["method"])
        if key not in SCORES:
            raise KeyError(f"missing audit score for {key}")
        scores = SCORES[key]
        audit = dict(source)
        for field, value in zip(SCORE_FIELDS, scores[:5]):
            audit[field] = str(value)
        audit["failure_mode"] = scores[5]
        audit["visual_note"] = scores[6]
        audit["audit_source"] = "manual review of e2_support_matched seed10/seed11 grids"
        rows.append(audit)
    return rows


def make_visual_summary(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in audit_rows:
        by_method[row["method"]].append(row)

    summary = []
    for method in METHOD_ORDER:
        rows = by_method[method]
        row = {
            "method": method,
            "n": str(len(rows)),
            "backbone": "SD3",
            "e2_layer": "E2.4 support-matched diagnostic",
        }
        for field in SCORE_FIELDS:
            row[field + "_mean"] = fmt(avg(rows, field))
        row["dominant_failure"] = dominant_failure(rows)
        summary.append(row)
    return summary


def dominant_failure(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        mode = row["failure_mode"]
        if mode != "success":
            counts[mode] += 1
    if not counts:
        return "none"
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def make_combined_table(
    metric_rows: list[dict[str, str]], visual_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    metric_by_method = {row["method"]: row for row in metric_rows}
    visual_by_method = {row["method"]: row for row in visual_rows}
    combined = []
    for method in METHOD_ORDER:
        metric = metric_by_method[method]
        visual = visual_by_method[method]
        row = {
            "method": method,
            "n": metric["n"],
            "backbone": "SD3",
            "e2_layer": "E2.4 support-matched diagnostic",
            "claim_boundary": metric["claim_boundary"],
        }
        for field in METRIC_FIELDS:
            row[field] = metric[field]
        for field in SCORE_FIELDS:
            row[field + "_mean"] = visual[field + "_mean"]
        row["dominant_failure"] = visual["dominant_failure"]
        combined.append(row)
    return combined


def write_markdown(path: Path, combined: list[dict[str, str]]) -> None:
    lines = [
        "# E2.4 Support-Matched Diagnostic With Visual Audit",
        "",
        "Scope: cat_crown, tshirt_star, and backpack_remove_toy_charm; seeds 10/11.",
        "This is the recommended compact support-matched subset: attached accessory addition, surface decal, and exposed-object removal.",
        "Post-hoc blend rows use the same fixed binary edit mask and are diagnostic only.",
        "",
        "| Method | n | Outside L1 down | SSIM up | Edit | Preserve | Locality | Artifact | Overall | Failure mode |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in combined:
        lines.append(
            f"| {row['method']} | {row['n']} | "
            f"{row['outside_mask_l1_mean']} | {row['source_ssim_luma_mean']} | "
            f"{row['edit_success_mean']} | {row['preservation_mean']} | "
            f"{row['locality_mean']} | {row['artifact_mean']} | "
            f"{row['overall_mean']} | {row['dominant_failure']} |"
        )
    lines.extend(
        [
            "",
            "Conclusion: fixed binary output blending almost eliminates outside-mask metric error by construction, but visual audit shows it does not solve target correctness or boundary coherence. DeCE-RF is the only row that consistently performs the intended operation in this support-matched diagnostic. The fixed-weight DeCE displacement variant is reported separately as a component ablation rather than as an E2.4 support baseline.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    exp = Path("experiments/support_v3_2026-06-02")
    audit_dir = exp / "visual_audit"
    manifest_path = exp / "e2_support_matched_diagnostic_manifest.csv"
    metrics_table_path = exp / "e2_support_matched_contextual_table.csv"

    audit_rows = make_audit_rows(read_csv(manifest_path))
    audit_path = audit_dir / "e2_support_matched_visual_audit_filled.csv"
    audit_fields = list(audit_rows[0])
    write_csv(audit_path, audit_rows, audit_fields)

    summary_rows = make_visual_summary(audit_rows)
    summary_path = audit_dir / "e2_support_matched_visual_audit_summary.csv"
    summary_fields = list(summary_rows[0])
    write_csv(summary_path, summary_rows, summary_fields)

    combined_rows = make_combined_table(read_csv(metrics_table_path), summary_rows)
    combined_csv = exp / "e2_support_matched_contextual_table_with_audit.csv"
    combined_md = exp / "e2_support_matched_contextual_table_with_audit.md"
    write_csv(combined_csv, combined_rows, list(combined_rows[0]))
    write_markdown(combined_md, combined_rows)

    conclusion_path = audit_dir / "e2_support_matched_visual_audit_conclusion.md"
    conclusion_path.write_text(
        "\n".join(
            [
                "# E2.4 Visual Audit Conclusion",
                "",
                "Manual review of the seed10/seed11 support-matched grids confirms that post-hoc mask blending improves preservation metrics but frequently misses the requested edit or introduces visible paste boundaries.",
                "DeCE-RF is stronger on target correctness and overall quality. Fixed DeCE displacement is treated as a separate component ablation, so E2.4 stays focused on whether localization alone explains the gain.",
                "",
                f"Audit CSV: {audit_path}",
                f"Summary CSV: {summary_path}",
                f"Combined table: {combined_md}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {audit_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {combined_csv}")
    print(f"wrote {combined_md}")
    print(f"wrote {conclusion_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
