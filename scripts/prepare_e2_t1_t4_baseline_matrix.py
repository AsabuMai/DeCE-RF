from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path


ROOT = Path.cwd()
EXP = ROOT / "experiments" / "support_v3_2026-06-02"
MANIFEST = EXP / "e2_t1_t4_formal_baseline_manifest.csv"
SRC_ROOT = ROOT / "outputs" / "baselines"
OUT = ROOT / "outputs" / "e2_t1_t4_baseline_matrix"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def normalized_metadata(row: dict[str, str], src: Path) -> dict:
    metadata = json.loads(src.read_text(encoding="utf-8"))
    if not metadata.get("image"):
        metadata["image"] = row.get("source_image") or row.get("image") or ""
    if not metadata.get("source_prompt"):
        metadata["source_prompt"] = row.get("source_prompt") or row.get("effective_source_prompt") or ""
    if not metadata.get("target_prompt"):
        metadata["target_prompt"] = row.get("target_prompt") or row.get("effective_target_prompt") or ""
    metadata["baseline"] = row.get("baseline", metadata.get("baseline", ""))
    metadata["task"] = row.get("task", metadata.get("task", ""))
    metadata["seed"] = str(row.get("seed", metadata.get("seed", ""))).removeprefix("seed_")
    return metadata


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    copied = []
    missing = []
    for row in rows:
        baseline = row["baseline"]
        task = row["task"]
        seed = str(row["seed"]).removeprefix("seed_")
        if row.get("status") != "complete":
            missing.append({**row, "mirror_status": "manifest_not_complete"})
            continue
        src = SRC_ROOT / baseline / task / f"seed_{seed}"
        dst = OUT / task / baseline / f"seed_{seed}"
        required = [src / "result.png", src / "metadata.json", src / "command.txt"]
        if not all(path.is_file() for path in required):
            missing.append({**row, "mirror_status": "source_required_file_missing"})
            continue
        link_or_copy(src / "result.png", dst / "result.png")
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "metadata.json").write_text(
            json.dumps(normalized_metadata(row, src / "metadata.json"), indent=2) + "\n",
            encoding="utf-8",
        )
        link_or_copy(src / "command.txt", dst / "command.txt")
        (dst / "stats.json").write_text(json.dumps({"steps": [], "note": "baseline stats stub"}) + "\n", encoding="utf-8")
        copied.append(
            {
                "baseline": baseline,
                "task": task,
                "seed": seed,
                "result": str((dst / "result.png").relative_to(ROOT)),
                "metadata": str((dst / "metadata.json").relative_to(ROOT)),
                "command": str((dst / "command.txt").relative_to(ROOT)),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    copied_path = EXP / "e2_t1_t4_baseline_matrix_manifest.csv"
    with copied_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(copied[0]) if copied else ["baseline", "task", "seed", "result", "metadata", "command"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(copied)
    missing_path = EXP / "e2_t1_t4_baseline_matrix_missing.csv"
    with missing_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(missing[0]) if missing else ["baseline", "task", "seed", "mirror_status"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing)
    print(f"mirrored={len(copied)} missing={len(missing)}")
    print(f"wrote {copied_path}")
    print(f"wrote {missing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
