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
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output, [image.width, image.height]


def find_result(results_dir: Path, seed: str) -> Path:
    seed_dir = results_dir / f"seed_{seed}"
    candidates = sorted(seed_dir.glob("*/target_0.png")) if seed_dir.exists() else []
    return candidates[-1] if candidates else seed_dir / "missing" / "target_0.png"


def complete_row(
    row: dict[str, str],
    *,
    repo_root: Path,
    run_dir: Path,
    task: str,
    seed: str,
    source_image_path: Path,
    image_path: Path,
    prepared_size: list[int],
    reflex_root: Path,
    reflex_result: Path,
    command_text: str,
) -> dict[str, str]:
    shutil.copy2(reflex_result, run_dir / "result.png")
    matched_conditions = (
        "official ReFlex FLUX-dev runner with local dynamic-latent compatibility patch; "
        "source image resized with max_image_size=512 before ReFlex processing; source prompt, target prompt, "
        "and seed match; automatic attention-mask mode; num_inference_steps=28, "
        "guidance_scale=3.5, feature_steps=5, attn_topk=20"
    )
    metadata = {
        "baseline": "reflex",
        "task": task,
        "seed": int(seed),
        "source_image": row["source_image"],
        "source_prompt": row["source_prompt"],
        "target_prompt": row["target_prompt"],
        "resolution": prepared_size,
        "matched_conditions": matched_conditions,
        "reflex_root": str(reflex_root),
        "original_source_image": str(source_image_path),
        "prepared_source_image": str(image_path),
        "reflex_output": str(reflex_result),
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
            "notes": "official ReFlex FLUX-dev runner",
        }
    )
    return row


def run_row(
    row: dict[str, str],
    *,
    repo_root: Path,
    reflex_root: Path,
    python_bin: str,
    device: str,
    max_image_size: int,
    dry_run: bool,
) -> dict[str, str]:
    task = row["task"]
    seed = row["seed"]
    run_dir = repo_root / "outputs" / "baselines" / "reflex" / task / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp_reflex"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    source_image_path = (repo_root / row["source_image"]).resolve()
    image_path, prepared_size = prepare_input_image(source_image_path, tmp_dir / "input_512.png", max_image_size)
    results_dir = tmp_dir / "results"

    cmd = [
        python_bin,
        str(reflex_root / "img_edit.py"),
        "--gpu",
        device,
        "--seed",
        str(seed),
        "--img_path",
        str(image_path),
        "--source_prompt",
        row["source_prompt"],
        "--target_prompt",
        row["target_prompt"],
        "--results_dir",
        str(results_dir),
        "--feature_steps",
        "5",
        "--attn_topk",
        "20",
    ]
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    command_text = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True " + " ".join(
        subprocess.list2cmdline([part]) for part in cmd
    )
    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    existing_result = find_result(results_dir, seed)
    if existing_result.exists() and not dry_run:
        return complete_row(
            row,
            repo_root=repo_root,
            run_dir=run_dir,
            task=task,
            seed=seed,
            source_image_path=source_image_path,
            image_path=image_path,
            prepared_size=prepared_size,
            reflex_root=reflex_root,
            reflex_result=existing_result,
            command_text=command_text,
        )
    if dry_run:
        row.update({"status": "pending", "command": str((run_dir / "command.txt").relative_to(repo_root)), "notes": "dry_run"})
        return row

    try:
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cmd,
                cwd=reflex_root,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, **env},
            )
        if completed.returncode != 0:
            raise RuntimeError(f"ReFlex exited with code {completed.returncode}")
        result = find_result(results_dir, seed)
        if not result.exists():
            raise FileNotFoundError(f"missing ReFlex output: {result}")
        return complete_row(
            row,
            repo_root=repo_root,
            run_dir=run_dir,
            task=task,
            seed=seed,
            source_image_path=source_image_path,
            image_path=image_path,
            prepared_size=prepared_size,
            reflex_root=reflex_root,
            reflex_result=result,
            command_text=command_text,
        )
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        (run_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "baseline": "reflex",
                    "task": task,
                    "seed": int(seed),
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
                "notes": "official ReFlex FLUX-dev runner",
            }
        )
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched ReFlex FLUX-dev baselines.")
    parser.add_argument("--manifest", default="experiments/baseline_parity_manifest.csv", type=Path)
    parser.add_argument("--reflex-root", default="/home/Wu_25R8111/ReFlex", type=Path)
    parser.add_argument("--python", default="/home/Wu_25R8111/ENTER/envs/flowedit/bin/python")
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
        if row.get("baseline") != "reflex":
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
            reflex_root=args.reflex_root,
            python_bin=args.python,
            device=args.device,
            max_image_size=args.max_image_size,
            dry_run=args.dry_run,
        )
        selected += 1
    write_manifest(args.manifest, rows)
    print(f"processed {selected} ReFlex row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
