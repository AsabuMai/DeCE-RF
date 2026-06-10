from __future__ import annotations

import csv
from pathlib import Path


TASKS = [
    "cat_crown",
    "bowl_apple_inside",
    "tshirt_star",
    "red_chair_blue",
    "pillow_vertical_fabric_strip",
    "backpack_remove_toy_charm",
]
METHODS = ["fireflow", "rf_solver_edit", "reflex"]
SEEDS = ["10", "11", "12"]


def main() -> int:
    root = Path.cwd()
    exp = root / "experiments" / "support_v3_2026-06-02"
    normalized_manifest = exp / "e2_native_flux_normalized_512_manifest.csv"
    out_csv = exp / "visual_audit" / "e2_native_flux_visual_audit_template.csv"
    out_md = exp / "visual_audit" / "e2_native_flux_visual_audit_readme.md"

    rows = list(csv.DictReader(normalized_manifest.open(newline="", encoding="utf-8")))
    lookup = {
        (row["method"], row["task"], row["seed"]): row
        for row in rows
        if row["group"] == "e2_native_flux_result"
    }

    out_rows = []
    for task in TASKS:
        for seed in SEEDS:
            for method in METHODS:
                row = lookup[(method, task, seed)]
                out_rows.append(
                    {
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "backbone": "FLUX.1-dev",
                        "normalized_result": row["normalized_path"],
                        "original_result": row["source_path"],
                        "grid": str(
                            exp
                            / "visual_audit"
                            / "e2_native_flux_grids"
                            / f"e2_native_flux_seed{seed}_grid.png"
                        ),
                        "edit_success_1_5": "",
                        "preservation_1_5": "",
                        "locality_1_5": "",
                        "artifact_1_5": "",
                        "overall_1_5": "",
                        "failure_flag": "",
                        "notes": "",
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0])
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    out_md.write_text(
        "\n".join(
            [
                "# E2 Native FLUX Visual Audit",
                "",
                "Score each native FLUX contextual row on 1-5 scales.",
                "",
                "- edit_success_1_5: requested edit is visible and semantically correct.",
                "- preservation_1_5: source identity/layout/background are preserved.",
                "- locality_1_5: change stays near the intended edit region.",
                "- artifact_1_5: fewer artifacts receives a higher score.",
                "- overall_1_5: paper-facing usefulness under the E2.3 contextual claim.",
                "",
                "These rows are E2.3 native implementation context only, not E2.2 same-backbone algorithmic evidence.",
                "",
                f"Template: `{out_csv}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
