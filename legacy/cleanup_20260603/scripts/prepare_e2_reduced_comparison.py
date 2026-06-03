#!/usr/bin/env python3
"""Normalize reduced E2 RF baseline outputs for fixed-mask metrics."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


TASKS = [
    "cat_crown",
    "bowl_apple_inside",
    "tshirt_star",
    "red_chair_blue",
    "pillow_vertical_fabric_strip",
    "backpack_remove_toy_charm",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_run(
    *,
    src_result: Path,
    src_metadata: Path | None,
    src_command: Path | None,
    out_dir: Path,
    source_image: str,
    source_prompt: str,
    target_prompt: str,
    extra_metadata: dict[str, object],
) -> bool:
    if not src_result.exists():
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_result, out_dir / "result.png")
    command_text = ""
    if src_command is not None and src_command.exists():
        command_text = src_command.read_text(encoding="utf-8", errors="replace")
    (out_dir / "command.txt").write_text(command_text or "command unavailable\n", encoding="utf-8")
    metadata = {
        "image": source_image,
        "source_prompt": source_prompt,
        "target_prompt": target_prompt,
        "e2_external_baseline": True,
        **extra_metadata,
    }
    if src_metadata is not None and src_metadata.exists():
        try:
            metadata["source_metadata"] = json.loads(src_metadata.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            metadata["source_metadata_error"] = repr(exc)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out_dir / "stats.json").write_text("[]\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/e2_rf_comparison"))
    parser.add_argument("--seed", default="10")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    selected = {
        (row["baseline"], row["task"], row["seed"]): row
        for row in rows
        if row.get("status") == "complete"
    }
    copied = []
    for task in TASKS:
        for baseline in ("flowedit", "flowalign", "splitflow"):
            row = selected.get((baseline, task, args.seed))
            if not row:
                continue
            ok = copy_run(
                src_result=Path(row["result_image"]),
                src_metadata=Path(row["metadata"]) if row.get("metadata") else None,
                src_command=Path(row["command"]) if row.get("command") else None,
                out_dir=args.out_dir / task / baseline / f"seed_{args.seed}",
                source_image=row["source_image"],
                source_prompt=row["source_prompt"],
                target_prompt=row["target_prompt"],
                extra_metadata={
                    "method": baseline,
                    "matched_conditions": row.get("matched_conditions", ""),
                    "baseline_manifest_row": row,
                },
            )
            if ok:
                copied.append((task, baseline))

        dece_dir = Path("outputs/pretty_matrix") / task / "support_v3_controller_rmsgap" / f"seed_{args.seed}"
        dece_meta = dece_dir / "metadata.json"
        if dece_meta.exists():
            meta = json.loads(dece_meta.read_text(encoding="utf-8"))
            fallback = (
                selected.get(("flowedit", task, args.seed))
                or selected.get(("flowalign", task, args.seed))
                or selected.get(("splitflow", task, args.seed))
            )
            source_image = meta.get("image") or meta.get("source_image") or (fallback["source_image"] if fallback else "")
            source_prompt = meta.get("source_prompt") or meta.get("effective_source_prompt") or (fallback["source_prompt"] if fallback else "")
            target_prompt = meta.get("target_prompt") or meta.get("effective_target_prompt") or (fallback["target_prompt"] if fallback else "")
            ok = copy_run(
                src_result=dece_dir / "result.png",
                src_metadata=dece_meta,
                src_command=dece_dir / "command.txt",
                out_dir=args.out_dir / task / "support_v3_controller_rmsgap" / f"seed_{args.seed}",
                source_image=source_image,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                extra_metadata={
                    "method": "support_v3_controller_rmsgap",
                    "matched_conditions": "internal DeCE-RF strict Phase 1 run",
                },
            )
            if ok:
                copied.append((task, "support_v3_controller_rmsgap"))

    print(f"copied {len(copied)} runs to {args.out_dir}")
    for task, method in copied:
        print(task, method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
