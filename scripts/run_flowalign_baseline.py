#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image


FIELDS = [
    "baseline",
    "task",
    "seed",
    "status",
    "source_image",
    "source_prompt",
    "target_prompt",
    "result_image",
    "metadata",
    "command",
    "matched_conditions",
    "failure_reason",
    "notes",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_input_image(source: Path, output: Path, max_image_size: int) -> tuple[Path, list[int]]:
    image = Image.open(source).convert("RGB")
    if max(image.size) > max_image_size:
        scale = max_image_size / float(max(image.size))
        image = image.resize(
            (max(16, int(round(image.width * scale))), max(16, int(round(image.height * scale)))),
            Image.Resampling.LANCZOS,
        )
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output, [image.width, image.height]


def run_row(
    row: dict[str, str],
    *,
    repo_root: Path,
    flowalign_root: Path,
    python_bin: str,
    device: str,
    max_image_size: int,
    dry_run: bool,
) -> dict[str, str]:
    task = row["task"]
    seed = row["seed"]
    run_dir = repo_root / "outputs" / "baselines" / "flowalign" / task / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp_flowalign"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    source_image_path = (repo_root / row["source_image"]).resolve()
    image_path, prepared_size = prepare_input_image(source_image_path, tmp_dir / "input_512.png", max_image_size)
    workdir = tmp_dir / "workdir"
    output_path = workdir / "edited" / image_path.name

    cmd = [
        python_bin,
        str(flowalign_root / "run_edit.py"),
        "--model",
        "sd3",
        "--method",
        "flowalign",
        "--img_path",
        str(image_path),
        "--src_prompt",
        row["source_prompt"],
        "--tgt_prompt",
        row["target_prompt"],
        "--img_shape",
        str(max_image_size),
        "--cfg_scale",
        "13.5",
        "--workdir",
        str(workdir),
        "--seed",
        str(seed),
        "--NFE",
        "33",
        "--n_start",
        "0",
        "--efficient_memory",
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": device,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": os.environ.get("HF_HOME", "/workspace/.cache/huggingface"),
    }
    command_text = (
        f"CUDA_VISIBLE_DEVICES={device} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        + " ".join(subprocess.list2cmdline([part]) for part in cmd)
    )
    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
    if dry_run:
        row.update({"status": "pending", "command": str((run_dir / "command.txt").relative_to(repo_root)), "notes": "dry_run"})
        return row

    try:
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cmd,
                cwd=flowalign_root,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, **env},
            )
        if completed.returncode != 0:
            raise RuntimeError(f"FlowAlign exited with code {completed.returncode}")
        if not output_path.exists():
            raise FileNotFoundError(f"missing FlowAlign output: {output_path}")
        shutil.copy2(output_path, run_dir / "result.png")

        matched_conditions = (
            "official FlowAlign SD3 runner in target-prompt mode; source image resized to max_image_size=512; "
            "source prompt, target prompt, seed, cfg_scale=13.5, NFE=33, n_start=0; efficient_memory enabled"
        )
        metadata = {
            "baseline": "flowalign",
            "task": task,
            "seed": int(seed),
            "image": str(source_image_path),
            "source_image": row["source_image"],
            "source_prompt": row["source_prompt"],
            "target_prompt": row["target_prompt"],
            "resolution": prepared_size,
            "matched_conditions": matched_conditions,
            "flowalign_root": str(flowalign_root),
            "prepared_source_image": str(image_path),
            "flowalign_output": str(output_path),
            "command": command_text,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        row.update(
            {
                "status": "complete",
                "result_image": str((run_dir / "result.png").relative_to(repo_root)),
                "metadata": str((run_dir / "metadata.json").relative_to(repo_root)),
                "command": str((run_dir / "command.txt").relative_to(repo_root)),
                "matched_conditions": matched_conditions,
                "failure_reason": "",
                "notes": "official FlowAlign SD3 target-mode runner",
            }
        )
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "baseline": "flowalign",
                    "task": task,
                    "seed": int(seed),
                    "image": str(source_image_path),
                    "source_image": row["source_image"],
                    "prepared_source_image": str(image_path),
                    "source_prompt": row["source_prompt"],
                    "target_prompt": row["target_prompt"],
                    "failure_reason": failure,
                    "command": command_text,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        row.update(
            {
                "status": "failed",
                "metadata": str((run_dir / "metadata.json").relative_to(repo_root)),
                "command": str((run_dir / "command.txt").relative_to(repo_root)),
                "failure_reason": failure,
                "notes": "official FlowAlign SD3 target-mode runner",
            }
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched FlowAlign SD3 target-mode baselines.")
    parser.add_argument("--manifest", default="experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv", type=Path)
    parser.add_argument("--flowalign-root", default="/workspace/baselines/src/FlowAlign", type=Path)
    parser.add_argument("--python", default="/workspace/baselines/envs/flowalign-py310/bin/python")
    parser.add_argument("--device", default="0")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo_root = Path.cwd()
    rows = read_manifest(args.manifest)
    task_filter = {item for item in args.tasks.split() if item}
    seed_filter = {item.removeprefix("seed_") for item in args.seeds.split() if item}
    selected = 0
    for index, row in enumerate(rows):
        if row.get("baseline") != "flowalign":
            continue
        if task_filter and row.get("task") not in task_filter:
            continue
        if seed_filter and row.get("seed") not in seed_filter:
            continue
        if args.skip_complete and row.get("status") == "complete":
            continue
        if args.limit and selected >= args.limit:
            break
        rows[index] = run_row(
            row,
            repo_root=repo_root,
            flowalign_root=args.flowalign_root,
            python_bin=args.python,
            device=args.device,
            max_image_size=args.max_image_size,
            dry_run=args.dry_run,
        )
        selected += 1
    write_manifest(args.manifest, rows)
    print(f"processed {selected} FlowAlign row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
