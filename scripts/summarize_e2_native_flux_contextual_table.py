from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


METRIC_COLUMNS = [
    "outside_mask_l1",
    "inside_mask_l1",
    "source_ssim_luma",
    "outside_mask_l1_excess_preserve_error",
]

VISUAL_COLUMNS = [
    "edit_success_1_5",
    "preservation_1_5",
    "locality_1_5",
    "artifact_1_5",
    "overall_1_5",
]


def avg(rows: list[dict[str, str]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, "")))
        except ValueError:
            pass
    return sum(values) / len(values) if values else 0.0


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> int:
    exp = Path("experiments/support_v3_2026-06-02")
    metrics = list(csv.DictReader((exp / "e2_native_flux_fixed_mask_metrics_with_context.csv").open(newline="", encoding="utf-8")))
    audit = list(csv.DictReader((exp / "visual_audit" / "e2_native_flux_visual_audit_filled.csv").open(newline="", encoding="utf-8")))
    out_csv = exp / "e2_native_flux_contextual_table.csv"
    out_md = exp / "e2_native_flux_contextual_table.md"

    by_method_metrics: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_method_audit: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        by_method_metrics[row["method"]].append(row)
    for row in audit:
        by_method_audit[row["method"]].append(row)

    methods = sorted(set(by_method_metrics) & set(by_method_audit))
    rows = []
    for method in methods:
        mrows = by_method_metrics[method]
        arows = by_method_audit[method]
        row = {
            "method": method,
            "backbone": "FLUX.1-dev",
            "n": str(len(mrows)),
            "resolution": "512x512 eval/display copy",
            "normalization_policy": "fit native output to 512x512 canvas; preserve aspect ratio; original retained",
            "e2_layer": "E2.3 native implementation context",
            "claim_boundary": "not E2.2 same-backbone algorithmic evidence",
        }
        for col in METRIC_COLUMNS:
            row[col + "_mean"] = fmt(avg(mrows, col))
        for col in VISUAL_COLUMNS:
            row[col + "_mean"] = fmt(avg(arows, col))
        rows.append(row)

    fieldnames = list(rows[0])
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# E2.3 Native FLUX Contextual Table",
        "",
        "Backbone differs from the SD3 algorithmic comparison. These rows are native implementation context only.",
        "All metrics use fixed Core-6 evaluation masks and a 512x512 eval/display copy; original native outputs are retained.",
        "",
        "| Method | n | Backbone | Outside L1 down | Inside L1 | Source SSIM up | Excess outside L1 down | Edit audit up | Preserve audit up | Locality audit up | Artifact audit up | Overall audit up |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['method']} | {row['n']} | {row['backbone']} | "
            f"{row['outside_mask_l1_mean']} | {row['inside_mask_l1_mean']} | "
            f"{row['source_ssim_luma_mean']} | {row['outside_mask_l1_excess_preserve_error_mean']} | "
            f"{row['edit_success_1_5_mean']} | {row['preservation_1_5_mean']} | "
            f"{row['locality_1_5_mean']} | {row['artifact_1_5_mean']} | {row['overall_1_5_mean']} |"
        )
    md.extend(
        [
            "",
            "Interpretation: RF-Solver-Edit has the best visual overall score among the native FLUX rows because it preserves layout better on localized insertion/decal tasks. ReFlex has stronger target formation but substantially worse preservation/locality due to broad re-rendering. FireFlow is polished and preserves some structure, but misses several Core-6 goals, especially recolor, strip editing, and removal.",
            "",
            "Paper wording: report this as E2.3 native FLUX implementation context, with backbone and normalization columns visible. Do not merge it into the E2.2 same-backbone SD3 algorithmic table.",
        ]
    )
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
