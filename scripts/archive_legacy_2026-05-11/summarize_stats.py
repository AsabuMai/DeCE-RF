from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def scalar(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    return float(value)


def avg(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(scalar(row, key) for row in rows) / len(rows)


def max_value(rows: list[dict], key: str) -> float:
    return max((scalar(row, key) for row in rows), default=0.0)


def summarize(path: Path) -> dict[str, str | float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    final = rows[-1] if rows else {}
    run_name = path.parent.name
    output_path = path.parent / "result.png"
    return {
        "run_name": run_name,
        "final_rec_energy": scalar(final, "rec_energy"),
        "final_edit_anchor_energy": scalar(final, "edit_anchor_energy"),
        "avg_rec_energy": avg(rows, "rec_energy"),
        "max_rec_energy": max_value(rows, "rec_energy"),
        "avg_rec_norm": avg(rows, "rec_guidance_norm"),
        "max_rec_norm": max_value(rows, "rec_guidance_norm"),
        "avg_trajectory_preserve_norm": avg(rows, "trajectory_preserve_norm"),
        "max_trajectory_preserve_norm": max_value(rows, "trajectory_preserve_norm"),
        "avg_edit_norm": avg(rows, "edit_guidance_norm"),
        "max_edit_norm": max_value(rows, "edit_guidance_norm"),
        "avg_total_velocity_norm": avg(rows, "total_velocity_norm"),
        "max_total_velocity_norm": max_value(rows, "total_velocity_norm"),
        "avg_beta_t": avg(rows, "beta_t"),
        "max_beta_t": max_value(rows, "beta_t"),
        "avg_cos_base_anchor": avg(rows, "cos_base_anchor"),
        "avg_cos_base_region": avg(rows, "cos_base_region"),
        "avg_cos_anchor_region": avg(rows, "cos_anchor_region"),
        "avg_cos_rec_base": avg(rows, "cos_rec_base"),
        "avg_cos_rec_edit_total": avg(rows, "cos_rec_edit_total"),
        "final_mask_area": scalar(final, "mask_area_ratio"),
        "avg_mask_area": avg(rows, "mask_area_ratio"),
        "final_core_mask_area": scalar(final, "core_mask_area_ratio"),
        "final_preserve_mask_area": scalar(final, "preserve_mask_area_ratio"),
        "mask_center_x": scalar(final, "mask_center_x"),
        "mask_center_y": scalar(final, "mask_center_y"),
        "mask_bbox_x0": scalar(final, "mask_bbox_x0"),
        "mask_bbox_y0": scalar(final, "mask_bbox_y0"),
        "mask_bbox_x1": scalar(final, "mask_bbox_x1"),
        "mask_bbox_y1": scalar(final, "mask_bbox_y1"),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RF h-Edit stats JSON files as CSV.")
    parser.add_argument("stats", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, help="Optional path to save the summaries as JSON.")
    args = parser.parse_args()

    summaries = [summarize(path) for path in args.stats]
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    fieldnames = [
        "run_name",
        "final_rec_energy",
        "final_edit_anchor_energy",
        "avg_rec_energy",
        "max_rec_energy",
        "avg_rec_norm",
        "max_rec_norm",
        "avg_trajectory_preserve_norm",
        "max_trajectory_preserve_norm",
        "avg_edit_norm",
        "max_edit_norm",
        "avg_total_velocity_norm",
        "max_total_velocity_norm",
        "avg_beta_t",
        "max_beta_t",
        "avg_cos_base_anchor",
        "avg_cos_base_region",
        "avg_cos_anchor_region",
        "avg_cos_rec_base",
        "avg_cos_rec_edit_total",
        "final_mask_area",
        "avg_mask_area",
        "final_core_mask_area",
        "final_preserve_mask_area",
        "mask_center_x",
        "mask_center_y",
        "mask_bbox_x0",
        "mask_bbox_y0",
        "mask_bbox_x1",
        "mask_bbox_y1",
        "output_path",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summaries)


if __name__ == "__main__":
    main()
