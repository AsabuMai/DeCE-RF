#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml
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


def find_flowedit_result(flowedit_root: Path, exp_name: str, image_path: Path, seed: str) -> Path:
    src_name = image_path.stem
    result_dir = flowedit_root / "outputs" / exp_name / "SD3" / f"src_{src_name}" / "tar_0"
    pattern = f"output_T_steps_28_n_avg_1_cfg_enc_3.5_cfg_dec10.5_n_min_0_n_max_24_seed{seed}.png"
    result = result_dir / pattern
    if not result.exists():
        candidates = sorted(result_dir.glob(f"*seed{seed}.png")) if result_dir.exists() else []
        if candidates:
            return candidates[-1]
    return result


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


def run_row(
    row: dict[str, str],
    *,
    repo_root: Path,
    flowedit_root: Path,
    python_bin: str,
    device: str,
    max_image_size: int,
    dry_run: bool,
) -> dict[str, str]:
    task = row["task"]
    seed = row["seed"]
    run_dir = repo_root / "outputs" / "baselines" / "flowedit" / task / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp_flowedit"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    source_image_path = (repo_root / row["source_image"]).resolve()
    image_path, prepared_size = prepare_input_image(
        source_image_path,
        tmp_dir / "input_512.png",
        max_image_size,
    )
    exp_name = f"matched_{task}_seed_{seed}"
    dataset_yaml = tmp_dir / "edits.yaml"
    exp_yaml = tmp_dir / "exp.yaml"

    dataset_payload = [
        {
            "input_img": str(image_path),
            "source_prompt": row["source_prompt"],
            "target_prompts": [row["target_prompt"]],
        }
    ]
    exp_payload = [
        {
            "exp_name": exp_name,
            "dataset_yaml": str(dataset_yaml),
            "model_type": "SD3",
            "sampler_type": "FlowEditSD3",
            "T_steps": 28,
            "n_avg": 1,
            "src_guidance_scale": 3.5,
            "tar_guidance_scale": 10.5,
            "n_min": 0,
            "n_max": 24,
            "seed": int(seed),
        }
    ]
    dataset_yaml.write_text(yaml.safe_dump(dataset_payload, sort_keys=False), encoding="utf-8")
    exp_yaml.write_text(yaml.safe_dump(exp_payload, sort_keys=False), encoding="utf-8")

    cmd = [
        python_bin,
        str(flowedit_root / "run_script.py"),
        "--device_number",
        "0",
        "--exp_yaml",
        str(exp_yaml),
    ]
    env = {"CUDA_VISIBLE_DEVICES": device}
    command_text = "CUDA_VISIBLE_DEVICES=" + device + " " + " ".join(subprocess.list2cmdline([part]) for part in cmd)
    (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")

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
        completed = subprocess.run(
            cmd,
            cwd=flowedit_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, **env},
        )
        (run_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"FlowEdit exited with code {completed.returncode}")

        flowedit_result = find_flowedit_result(flowedit_root, exp_name, image_path, seed)
        if not flowedit_result.exists():
            raise FileNotFoundError(f"missing FlowEdit output: {flowedit_result}")
        shutil.copy2(flowedit_result, run_dir / "result.png")

        metadata = {
            "baseline": "flowedit",
            "task": task,
            "seed": int(seed),
            "source_image": row["source_image"],
            "source_prompt": row["source_prompt"],
            "target_prompt": row["target_prompt"],
            "resolution": prepared_size,
            "matched_conditions": "source image resized with max_image_size=512 and cropped to multiples of 16, source prompt, target prompt, seed, SD3 model, T_steps=28, n_max=24, tar_guidance_scale=10.5",
            "flowedit_root": str(flowedit_root),
            "original_source_image": str(source_image_path),
            "prepared_source_image": str(image_path),
            "flowedit_output": str(flowedit_result),
            "command": command_text,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        row.update(
            {
                "status": "complete",
                "result_image": str((run_dir / "result.png").relative_to(repo_root)),
                "metadata": str((run_dir / "metadata.json").relative_to(repo_root)),
                "command": str((run_dir / "command.txt").relative_to(repo_root)),
                "matched_conditions": metadata["matched_conditions"],
                "failure_reason": "",
                "notes": "official FlowEdit SD3 runner",
            }
        )
    except Exception as exc:  # noqa: BLE001 - failure is recorded in the manifest.
        failure = f"{type(exc).__name__}: {exc}"
        metadata = {
            "baseline": "flowedit",
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
                "notes": "official FlowEdit SD3 runner",
            }
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched FlowEdit SD3 baselines from the parity manifest.")
    parser.add_argument("--manifest", default="experiments/baseline_parity_manifest.csv", type=Path)
    parser.add_argument("--flowedit-root", default="/home/Wu_25R8111/FlowEdit", type=Path)
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
        if row.get("baseline") != "flowedit":
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
            flowedit_root=args.flowedit_root,
            python_bin=args.python,
            device=args.device,
            max_image_size=args.max_image_size,
            dry_run=args.dry_run,
        )
        selected += 1

    write_manifest(args.manifest, rows)
    print(f"processed {selected} FlowEdit row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
