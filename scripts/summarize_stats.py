from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def avg(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def summarize(path: Path) -> dict[str, str | float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    final = rows[-1] if rows else {}
    run_name = path.parent.name
    output_path = path.parent / "result.png"
    return {
        "run_name": run_name,
        "final_rec_energy": float(final.get("rec_energy", 0.0)),
        "final_edit_anchor_energy": float(final.get("edit_anchor_energy", 0.0)),
        "avg_rec_norm": avg(rows, "rec_guidance_norm"),
        "max_rec_norm": max((float(row.get("rec_guidance_norm", 0.0)) for row in rows), default=0.0),
        "avg_trajectory_preserve_norm": avg(rows, "trajectory_preserve_norm"),
        "max_trajectory_preserve_norm": max(
            (float(row.get("trajectory_preserve_norm", 0.0)) for row in rows),
            default=0.0,
        ),
        "avg_edit_norm": avg(rows, "edit_guidance_norm"),
        "max_edit_norm": max((float(row.get("edit_guidance_norm", 0.0)) for row in rows), default=0.0),
        "avg_cos_base_anchor": avg(rows, "cos_base_anchor"),
        "avg_cos_rec_edit_total": avg(rows, "cos_rec_edit_total"),
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
        "avg_rec_norm",
        "max_rec_norm",
        "avg_trajectory_preserve_norm",
        "max_trajectory_preserve_norm",
        "avg_edit_norm",
        "max_edit_norm",
        "avg_cos_base_anchor",
        "avg_cos_rec_edit_total",
        "output_path",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summaries)


if __name__ == "__main__":
    main()
