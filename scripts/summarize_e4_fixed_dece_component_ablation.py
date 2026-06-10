from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = [
    "direct_target_raw",
    "support_v3_fixed",
    "support_v3_controller_rmsgap",
]

DISPLAY_NAME = {
    "direct_target_raw": "Direct target guidance",
    "support_v3_fixed": "Fixed DeCE displacement",
    "support_v3_controller_rmsgap": "DeCE-RF",
}

METRIC_FIELDS = [
    "outside_mask_l1",
    "inside_mask_l1",
    "source_ssim_luma",
    "outside_mask_l1_excess_preserve_error",
]

SCORE_FIELDS = [
    "edit_success",
    "preservation",
    "locality",
    "artifact",
    "overall",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def avg(rows: list[dict[str, str]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, "")))
        except ValueError:
            pass
    return sum(values) / len(values) if values else 0.0


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def fmt2(value: float) -> str:
    return f"{value:.2f}"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    exp = Path("experiments/support_v3_2026-06-02")
    metrics_path = exp / "e2_support_matched_fixed_mask_metrics.csv"
    audit_path = exp / "visual_audit" / "e2_support_matched_visual_audit_filled.csv"
    out_csv = exp / "e4_fixed_dece_component_ablation_compact.csv"
    out_md = exp / "e4_fixed_dece_component_ablation_compact.md"

    metric_rows = read_csv(metrics_path)
    audit_rows = read_csv(audit_path)
    metrics_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    audit_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in metric_rows:
        if row["method"] in METHOD_ORDER:
            metrics_by_method[row["method"]].append(row)
    for row in audit_rows:
        if row["method"] in METHOD_ORDER:
            audit_by_method[row["method"]].append(row)

    summary = []
    for method in METHOD_ORDER:
        rows = metrics_by_method[method]
        audit = audit_by_method[method]
        if not rows:
            raise RuntimeError(f"missing metrics for {method}")
        if not audit:
            raise RuntimeError(f"missing visual audit for {method}")
        row = {
            "method": method,
            "display_name": DISPLAY_NAME[method],
            "n": str(len(rows)),
            "backbone": "SD3",
            "scope": "compact component ablation on cat_crown, tshirt_star, backpack_remove_toy_charm; seeds 10/11",
            "claim_boundary": "component ablation; not an external baseline and not an E2.4 support-only row",
        }
        for field in METRIC_FIELDS:
            row[field + "_mean"] = fmt4(avg(rows, field))
        for field in SCORE_FIELDS:
            row[field + "_mean"] = fmt2(avg(audit, field))
        summary.append(row)

    write_csv(out_csv, summary, list(summary[0]))

    lines = [
        "# E4 Compact Component Ablation",
        "",
        "Scope: cat_crown, tshirt_star, and backpack_remove_toy_charm; seeds 10/11.",
        "`Fixed DeCE displacement` keeps operation-conditioned support and fixed clean-estimate edit-preserve displacement, but removes feedback-updated weights and projection/correction. It is a component ablation, not an external baseline or an E2.4 support-only row.",
        "",
        "| Variant | n | Outside L1 down | SSIM up | Edit | Preserve | Locality | Artifact | Overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['display_name']} | {row['n']} | "
            f"{row['outside_mask_l1_mean']} | {row['source_ssim_luma_mean']} | "
            f"{row['edit_success_mean']} | {row['preservation_mean']} | "
            f"{row['locality_mean']} | {row['artifact_mean']} | {row['overall_mean']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this compact ablation separates direct target guidance, fixed DeCE displacement, and the full DeCE-RF feedback controller. The fixed row should be used to discuss component structure and feedback, while E2.4 remains focused on whether binary localization alone explains the gain.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
