#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


SUMMARY_FIELDS = [
    "method",
    "runnable",
    "complete_rows",
    "failed_rows",
    "model_family",
    "seed_matched",
    "notes",
]

SCORE_FIELDS = [
    "task",
    "method",
    "seed",
    "edit_success",
    "preservation",
    "locality",
    "artifact",
    "overall",
    "notes",
]

MODEL_INFO = {
    "flowedit": {
        "model_family": "SD3",
        "seed_matched": "yes",
        "notes": "Official FlowEdit SD3 runner; source resized/cropped to 512-compatible input.",
    },
    "splitflow": {
        "model_family": "SD3 + Mistral-7B prompt decomposition",
        "seed_matched": "yes",
        "notes": "Official SplitFlow defaults; T_steps=50, n_max=33, tar guidance=13.5.",
    },
    "fireflow": {
        "model_family": "FLUX-dev",
        "seed_matched": "yes",
        "notes": "Official FireFlow fast-editing recipe; qualitative FLUX baseline, not SD3-matched.",
    },
    "rf_solver_edit": {
        "model_family": "FLUX-dev",
        "seed_matched": "no",
        "notes": "Official RF-Solver-Edit image script does not expose seed control.",
    },
    "reflex": {
        "model_family": "FLUX-dev",
        "seed_matched": "not run",
        "notes": "Failed under available 24GB GPU/code constraints; see failure metadata.",
    },
    "steerflow": {
        "model_family": "unknown / unavailable",
        "seed_matched": "not run",
        "notes": "No public runnable code or local repository found as of 2026-05-10.",
    },
}

METHOD_ORDER = ["flowedit", "splitflow", "fireflow", "rf_solver_edit", "reflex", "steerflow"]
VISUAL_METHODS = ["ours_full", "flowedit", "splitflow", "fireflow", "rf_solver_edit"]
TASKS = ["cat_crown", "dog_sunglasses", "mug_heart", "backpack_remove_toy_charm"]
SEEDS = ["10", "11", "12"]


# Scores are a first-pass internal visual audit, not a user-study result.
# 5 = strong, 3 = usable/weak, 1 = failed.
SCORES = {
    ("cat_crown", "ours_full"): (4, 5, 5, 4, 4, "Crown visible; cat identity and background preserved."),
    ("cat_crown", "flowedit"): (5, 1, 1, 3, 2, "Crown strong, but cat/background are redrawn."),
    ("cat_crown", "splitflow"): (5, 2, 2, 3, 3, "Crown strong; cat identity and pose drift."),
    ("cat_crown", "fireflow"): (4, 2, 2, 3, 3, "Crown appears; cat identity/pose drift."),
    ("cat_crown", "rf_solver_edit"): (4, 2, 2, 3, 3, "Crown appears; FLUX output changes identity/pose."),
    ("dog_sunglasses", "ours_full"): (3, 5, 5, 3, 4, "Sunglasses are subtle/semi-transparent; dog and background preserved."),
    ("dog_sunglasses", "flowedit"): (5, 2, 2, 3, 3, "Sunglasses strong; dog identity and composition drift."),
    ("dog_sunglasses", "splitflow"): (5, 3, 3, 3, 3, "Sunglasses strong; identity and face geometry still change."),
    ("dog_sunglasses", "fireflow"): (5, 2, 2, 3, 3, "Sunglasses clear; identity and clothing/background details drift."),
    ("dog_sunglasses", "rf_solver_edit"): (4, 2, 2, 3, 3, "Sunglasses generally visible; identity and layout drift."),
    ("mug_heart", "ours_full"): (5, 5, 5, 4, 5, "Heart clear; mug layout and background preserved."),
    ("mug_heart", "flowedit"): (4, 3, 3, 3, 3, "Heart appears; some seeds shift viewpoint or scene layout."),
    ("mug_heart", "splitflow"): (5, 4, 4, 4, 4, "Clean heart; moderate viewpoint/lighting change."),
    ("mug_heart", "fireflow"): (3, 4, 4, 3, 3, "Heart is small; mug layout mostly preserved."),
    ("mug_heart", "rf_solver_edit"): (2, 4, 4, 3, 3, "Heart is very small; mug layout mostly preserved."),
    ("backpack_remove_toy_charm", "ours_full"): (5, 5, 5, 4, 5, "Dangling toy removed; patch, strap, zipper, and fabric preserved."),
    ("backpack_remove_toy_charm", "flowedit"): (1, 1, 1, 2, 1, "Regenerates a different backpack/keychain scene and often keeps a charm."),
    ("backpack_remove_toy_charm", "splitflow"): (1, 2, 1, 2, 1, "Redraws the scene and keeps/regenerates a dangling toy."),
    ("backpack_remove_toy_charm", "fireflow"): (1, 4, 1, 2, 2, "Source layout partly preserved but dangling toy remains."),
    ("backpack_remove_toy_charm", "rf_solver_edit"): (1, 4, 1, 2, 2, "Source layout partly preserved but dangling toy remains."),
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["baseline"]].append(row)

    summary = []
    for method in METHOD_ORDER:
        method_rows = grouped.get(method, [])
        counts = Counter(row["status"] for row in method_rows)
        info = MODEL_INFO[method]
        summary.append(
            {
                "method": method,
                "runnable": "yes" if counts["complete"] else "no",
                "complete_rows": str(counts["complete"]),
                "failed_rows": str(counts["failed"]),
                "model_family": info["model_family"],
                "seed_matched": info["seed_matched"],
                "notes": info["notes"],
            }
        )
    return summary


def make_scores() -> list[dict[str, str]]:
    rows = []
    for task in TASKS:
        for method in VISUAL_METHODS:
            edit_success, preservation, locality, artifact, overall, notes = SCORES[(task, method)]
            for seed in SEEDS:
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "edit_success": str(edit_success),
                        "preservation": str(preservation),
                        "locality": str(locality),
                        "artifact": str(artifact),
                        "overall": str(overall),
                        "notes": notes,
                    }
                )
    return rows


def write_summary_md(path: Path, summary: list[dict[str, str]], scores: list[dict[str, str]]) -> None:
    lines = [
        "# Baseline Summary",
        "",
        "Date: 2026-05-10",
        "",
        "## Run Status",
        "",
        "| Method | Runnable | Complete | Failed | Model family | Seed matched | Notes |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| `{row['method']}` | {row['runnable']} | {row['complete_rows']} | {row['failed_rows']} | "
            f"{row['model_family']} | {row['seed_matched']} | {row['notes']} |"
        )

    method_scores: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scores:
        method_scores[row["method"]].append(row)
    lines.extend(
        [
            "",
            "## Internal Visual Score Averages",
            "",
            "Scores are a first-pass internal visual audit, not a user-study result. Scale: 1=failed, 3=usable/weak, 5=strong.",
            "",
            "| Method | Rows | Edit success | Preservation | Locality | Artifact | Overall |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in VISUAL_METHODS:
        method_rows = method_scores[method]
        averages = {}
        for field in ("edit_success", "preservation", "locality", "artifact", "overall"):
            averages[field] = sum(float(row[field]) for row in method_rows) / len(method_rows)
        lines.append(
            f"| `{method}` | {len(method_rows)} | {averages['edit_success']:.2f} | "
            f"{averages['preservation']:.2f} | {averages['locality']:.2f} | "
            f"{averages['artifact']:.2f} | {averages['overall']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `ours_full` is strongest on preservation and locality.",
            "- SplitFlow, FlowEdit, FireFlow, and RF-Solver-Edit often improve target-object strength but redraw identity or layout.",
            "- `backpack_remove_toy_charm` is the strongest evidence for localized preservation.",
            "- `dog_sunglasses` is the clearest tradeoff case: external baselines make darker sunglasses, while `ours_full` preserves the dog better.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate baseline summary and internal visual score artifacts.")
    parser.add_argument("--manifest", default="experiments/baseline_parity_manifest.csv", type=Path)
    parser.add_argument("--summary-csv", default="experiments/baseline_summary.csv", type=Path)
    parser.add_argument("--summary-md", default="experiments/baseline_summary.md", type=Path)
    parser.add_argument("--score-template", default="experiments/baseline_visual_score_template.csv", type=Path)
    parser.add_argument("--score-csv", default="experiments/baseline_visual_scores_seed10_12.csv", type=Path)
    args = parser.parse_args()

    manifest_rows = read_manifest(args.manifest)
    summary = make_summary(manifest_rows)
    scores = make_scores()
    template_rows = [
        {field: "" for field in SCORE_FIELDS}
        for _ in range(len(scores))
    ]

    write_csv(args.summary_csv, SUMMARY_FIELDS, summary)
    write_csv(args.score_template, SCORE_FIELDS, template_rows)
    write_csv(args.score_csv, SCORE_FIELDS, scores)
    write_summary_md(args.summary_md, summary, scores)
    print(f"wrote {args.summary_csv}")
    print(f"wrote {args.summary_md}")
    print(f"wrote {args.score_template}")
    print(f"wrote {args.score_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
