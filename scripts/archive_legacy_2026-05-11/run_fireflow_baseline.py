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
        resized_w = max(16, int(round(image.width * scale)))
        resized_h = max(16, int(round(image.height * scale)))
        image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output, [image.width, image.height]


def find_fireflow_result(output_dir: Path, prefix: str) -> Path:
    candidates = sorted(output_dir.glob(f"{prefix}_inject_1_start_layer_index_0_end_layer_index_37_img_*.jpg"))
    if not candidates:
        candidates = sorted(output_dir.glob("*.jpg"))
    if not candidates:
        return output_dir / f"{prefix}_inject_1_start_layer_index_0_end_layer_index_37_img_0.jpg"
    return candidates[-1]


def run_row(
    row: dict[str, str],
    *,
    repo_root: Path,
    fireflow_root: Path,
    python_bin: str,
    device: str,
    max_image_size: int,
    dry_run: bool,
) -> dict[str, str]:
    task = row["task"]
    seed = row["seed"]
    run_dir = repo_root / "outputs" / "baselines" / "fireflow" / task / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp_fireflow"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    source_image_path = (repo_root / row["source_image"]).resolve()
    image_path, prepared_size = prepare_input_image(
        source_image_path,
        tmp_dir / "input_512.png",
        max_image_size,
    )
    prefix = f"fireflow_{task}_seed_{seed}"
    fireflow_output_dir = tmp_dir / "fireflow_output"
    feature_dir = tmp_dir / "feature"

    cmd = [
        python_bin,
        str(fireflow_root / "src" / "edit.py"),
        "--source_prompt",
        row["source_prompt"],
        "--target_prompt",
        row["target_prompt"],
        "--guidance",
        "2",
        "--source_img_dir",
        str(image_path),
        "--num_steps",
        "8",
        "--inject",
        "1",
        "--start_layer_index",
        "0",
        "--end_layer_index",
        "37",
        "--name",
        "flux-dev",
        "--sampling_strategy",
        "fireflow",
        "--output_prefix",
        prefix,
        "--output_dir",
        str(fireflow_output_dir),
        "--feature_path",
        str(feature_dir),
        "--offload",
        "--seed",
        str(seed),
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": device,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    command_text = (
        "CUDA_VISIBLE_DEVICES="
        + device
        + " PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        + " ".join(subprocess.list2cmdline([part]) for part in cmd)
    )
    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

    existing_result = find_fireflow_result(fireflow_output_dir, prefix)
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
            fireflow_root=fireflow_root,
            fireflow_result=existing_result,
            command_text=command_text,
        )

    if dry_run:
        row.update(
            {
                "status": "pending",
                "command": str((run_dir / "command.txt").relative_to(repo_root)),
                "notes": "dry_run",
            }
        )
        return row

    try:
        with (run_dir / "run.log").open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cmd,
                cwd=fireflow_root / "src",
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, **env},
            )
        if completed.returncode != 0:
            raise RuntimeError(f"FireFlow exited with code {completed.returncode}")

        fireflow_result = find_fireflow_result(fireflow_output_dir, prefix)
        if not fireflow_result.exists():
            raise FileNotFoundError(f"missing FireFlow output: {fireflow_result}")
        return complete_row(
            row,
            repo_root=repo_root,
            run_dir=run_dir,
            task=task,
            seed=seed,
            source_image_path=source_image_path,
            image_path=image_path,
            prepared_size=prepared_size,
            fireflow_root=fireflow_root,
            fireflow_result=fireflow_result,
            command_text=command_text,
        )
    except Exception as exc:  # noqa: BLE001 - failure is recorded in the manifest.
        failure = f"{type(exc).__name__}: {exc}"
        metadata = {
            "baseline": "fireflow",
            "task": task,
            "seed": int(seed),
            "source_image": row["source_image"],
            "prepared_source_image": str(image_path),
            "source_prompt": row["source_prompt"],
            "target_prompt": row["target_prompt"],
            "failure_reason": failure,
            "command": command_text,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        row.update(
            {
                "status": "failed",
                "metadata": str((run_dir / "metadata.json").relative_to(repo_root)),
                "command": str((run_dir / "command.txt").relative_to(repo_root)),
                "failure_reason": failure,
                "notes": "official FireFlow FLUX-dev runner",
            }
        )
    return row


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
    fireflow_root: Path,
    fireflow_result: Path,
    command_text: str,
) -> dict[str, str]:
    Image.open(fireflow_result).convert("RGB").save(run_dir / "result.png")

    matched_conditions = (
        "official FireFlow FLUX-dev runner; source image resized with max_image_size=512 "
        "and cropped to multiples of 16; source prompt, target prompt, and seed match; "
        "official fast-editing parameters num_steps=8, inject=1, guidance=2, "
        "start_layer_index=0, end_layer_index=37, sampling_strategy=fireflow, offload=True"
    )
    metadata = {
        "baseline": "fireflow",
        "task": task,
        "seed": int(seed),
        "source_image": row["source_image"],
        "source_prompt": row["source_prompt"],
        "target_prompt": row["target_prompt"],
        "resolution": prepared_size,
        "matched_conditions": matched_conditions,
        "fireflow_root": str(fireflow_root),
        "original_source_image": str(source_image_path),
        "prepared_source_image": str(image_path),
        "fireflow_output": str(fireflow_result),
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
            "notes": "official FireFlow FLUX-dev runner",
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched FireFlow FLUX-dev baselines from the parity manifest.")
    parser.add_argument("--manifest", default="experiments/baseline_parity_manifest.csv", type=Path)
    parser.add_argument("--fireflow-root", default="/home/Wu_25R8111/FireFlow", type=Path)
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
        if row.get("baseline") != "fireflow":
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
            fireflow_root=args.fireflow_root,
            python_bin=args.python,
            device=args.device,
            max_image_size=args.max_image_size,
            dry_run=args.dry_run,
        )
        selected += 1

    write_manifest(args.manifest, rows)
    print(f"processed {selected} FireFlow row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
