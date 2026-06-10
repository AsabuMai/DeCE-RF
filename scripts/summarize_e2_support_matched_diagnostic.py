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

METRICS = [
    "outside_mask_l1",
    "inside_mask_l1",
    "source_ssim_luma",
    "outside_mask_l1_excess_preserve_error",
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
    metrics_path = exp / "e2_support_matched_fixed_mask_metrics.csv"
    out_csv = exp / "e2_support_matched_contextual_table.csv"
    out_md = exp / "e2_support_matched_contextual_table.md"
    rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8")))
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    summary = []
    for method in METHOD_ORDER:
        method_rows = by_method[method]
        row = {
            "method": method,
            "n": str(len(method_rows)),
            "backbone": "SD3",
            "e2_layer": "E2.4 support-matched diagnostic",
            "support_condition": method_rows[0].get("support_condition", "") if method_rows else "",
            "claim_boundary": "diagnostic only; post-hoc blending rows are not main fair baselines",
        }
        for metric in METRICS:
            row[metric + "_mean"] = fmt(avg(method_rows, metric))
        summary.append(row)

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    md = [
        "# E2.4 Support-Matched Diagnostic",
        "",
        "Scope: cat_crown, tshirt_star, and backpack_remove_toy_charm; seeds 10/11.",
        "The post-hoc blend rows use the same fixed binary edit mask for localization only:",
        "`output = M_edit * edited_output + (1 - M_edit) * source`.",
        "These rows are diagnostic and must not be presented as main fair baselines.",
        "",
        "| Method | n | Backbone | Outside L1 down | Inside L1 | Source SSIM up | Excess outside L1 down |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        md.append(
            f"| {row['method']} | {row['n']} | {row['backbone']} | "
            f"{row['outside_mask_l1_mean']} | {row['inside_mask_l1_mean']} | "
            f"{row['source_ssim_luma_mean']} | {row['outside_mask_l1_excess_preserve_error_mean']} |"
        )
    md.extend(
        [
            "",
            "Interpretation: post-hoc localization sharply reduces outside-mask drift for direct-target and FlowEdit-style outputs, but it cannot recover missed edits, boundary coherence, or controller behavior. Full DeCE-RF is included as the target method; the fixed-weight DeCE displacement variant is reported separately as a component ablation rather than as an E2.4 support baseline.",
            "",
            "Paper-safe wording: E2.4 separates localization from controller design. Binary localization or output blending can improve preservation metrics, but it is a diagnostic transformation rather than an editing algorithm under matched inference conditions.",
        ]
    )
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
