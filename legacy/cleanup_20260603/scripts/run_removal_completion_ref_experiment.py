#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_TASKS = (
    "backpack_remove_toy_charm",
    "laptop_remove_sticker",
    "fridge_remove_yellow_magnet",
    "fridge_remove_peach_magnet",
    "whiteboard_remove_yellow_letter",
    "dog_remove_tennis_ball",
)
DEFAULT_OUTPUT_METHOD = "support_v3_controller_rmsgap_completion_ref"


def split_items(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def parse_recorded_command(path: Path) -> tuple[dict[str, str], list[str]]:
    tokens = shlex.split(path.read_text(encoding="utf-8").strip(), posix=True)
    env: dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        key, sep, val = tokens[idx].partition("=")
        if sep != "=" or not key.replace("_", "").isalnum() or tokens[idx].startswith("-"):
            break
        env[key] = val
        idx += 1
    if idx >= len(tokens):
        raise ValueError(f"No executable command found in {path}")
    return env, tokens[idx:]


def command_get(cmd: list[str], option: str) -> str | None:
    try:
        idx = cmd.index(option)
    except ValueError:
        return None
    if idx + 1 >= len(cmd):
        return None
    return cmd[idx + 1]


def command_set(cmd: list[str], option: str, value: str | Path | float | int) -> None:
    text = str(value)
    try:
        idx = cmd.index(option)
    except ValueError:
        cmd.extend([option, text])
        return
    if idx + 1 >= len(cmd) or cmd[idx + 1].startswith("--"):
        cmd.insert(idx + 1, text)
    else:
        cmd[idx + 1] = text


def shell_join(env: dict[str, str], cmd: list[str]) -> str:
    env_tokens = [f"{key}={shlex.quote(str(value))}" for key, value in env.items()]
    return " ".join(env_tokens + [shlex.quote(str(part)) for part in cmd])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_reference(args: argparse.Namespace, task: str, seed: str, ref_dir: Path) -> None:
    result = ref_dir / "result.png"
    mask = ref_dir / "masks" / "same_support_inpaint_mask.png"
    if result.exists() and mask.exists() and not args.force_reference:
        return

    command = [
        sys.executable,
        str(args.root / "scripts" / "same_support_inpaint_baseline.py"),
        "--root",
        str(args.root),
        "--tasks",
        task,
        "--seed",
        seed,
        "--source-method",
        args.source_method,
        "--method",
        args.completion_method,
        "--output-method",
        args.reference_method,
        "--radius",
        str(args.inpaint_radius),
        "--dilate",
        str(args.inpaint_dilate),
        "--blur",
        str(args.inpaint_blur),
        "--threshold",
        str(args.inpaint_threshold),
    ]
    print(shell_join({}, command))
    if not args.dry_run:
        subprocess.run(command, cwd=args.root, check=True)


def build_task_command(args: argparse.Namespace, task: str, seed: str) -> tuple[Path, dict[str, str], list[str]]:
    source_dir = args.root / "outputs" / "pretty_matrix" / task / args.source_method / f"seed_{seed}"
    source_command = source_dir / "command.txt"
    source_metadata = source_dir / "metadata.json"
    if not source_command.exists():
        raise FileNotFoundError(f"Missing source command: {source_command}")
    if not source_metadata.exists():
        raise FileNotFoundError(f"Missing source metadata: {source_metadata}")

    out_dir = args.root / "outputs" / "pretty_matrix" / task / args.output_method / f"seed_{seed}"
    ref_dir = args.root / "outputs" / "pretty_matrix" / task / args.reference_method / f"seed_{seed}"
    ref_image = ref_dir / "result.png"
    ref_mask = ref_dir / "masks" / "same_support_inpaint_mask.png"
    if not ref_image.exists() or not ref_mask.exists():
        if args.dry_run:
            print(f"dry-run missing reference placeholder: {ref_dir}")
        else:
            raise FileNotFoundError(f"Missing completion reference for {task} seed {seed}: {ref_dir}")

    env, cmd = parse_recorded_command(source_command)
    metadata = load_json(source_metadata)
    source_image = metadata.get("image") or command_get(cmd, "--image")
    if not source_image:
        raise ValueError(f"Cannot resolve source image for {task} seed {seed}")

    env["TASK"] = task
    env["METHOD"] = args.output_method
    env["SEED"] = seed

    command_set(cmd, "--output", out_dir / "result.png")
    command_set(cmd, "--stats-output", out_dir / "stats.json")
    command_set(cmd, "--metadata-output", out_dir / "metadata.json")
    command_set(cmd, "--mask-output-dir", out_dir / "masks")

    if args.guidance_mode in {"image_ref", "both"}:
        command_set(cmd, "--edit-ref-guidance-scale", args.edit_ref_guidance_scale)
        command_set(cmd, "--edit-ref-image", ref_image)
        command_set(cmd, "--edit-ref-mask", ref_mask)
        if args.edit_ref_structure_mode == "source":
            command_set(cmd, "--edit-ref-structure-image", source_image)
        elif args.edit_ref_structure_mode == "reference":
            command_set(cmd, "--edit-ref-structure-image", ref_image)
        command_set(cmd, "--edit-ref-chroma-mode", args.edit_ref_chroma_mode)
        command_set(cmd, "--edit-ref-luma-preserve-scale", args.edit_ref_luma_preserve_scale)
        command_set(cmd, "--edit-ref-gradient-preserve-scale", args.edit_ref_gradient_preserve_scale)
        command_set(cmd, "--edit-ref-smooth-kernel", args.edit_ref_smooth_kernel)
    else:
        command_set(cmd, "--edit-ref-guidance-scale", 0.0)
    if args.guidance_mode in {"clean_delta", "both"}:
        command_set(cmd, "--completion-clean-delta-scale", args.completion_clean_delta_scale)
        command_set(cmd, "--completion-clean-delta-image", ref_image)
        command_set(cmd, "--completion-clean-delta-mask", ref_mask)
        command_set(cmd, "--completion-clean-delta-schedule-start", args.completion_clean_delta_schedule_start)
        command_set(cmd, "--completion-clean-delta-schedule-stop", args.completion_clean_delta_schedule_stop)
        command_set(cmd, "--completion-clean-delta-schedule-power", args.completion_clean_delta_schedule_power)
    command_set(cmd, "--edit-text-guidance-scale", args.edit_text_guidance_scale)

    if args.with_clean_fill:
        command_set(cmd, "--removal-controller-mode", "clean_fill")
        command_set(cmd, "--removal-fill-scale", args.removal_fill_scale)
        command_set(cmd, "--removal-suppression-scale", args.removal_suppression_scale)
        command_set(cmd, "--removal-ring-rec-scale", args.removal_ring_rec_scale)
    else:
        command_set(cmd, "--removal-controller-mode", "none")

    return out_dir, env, cmd


def write_protocol(args: argparse.Namespace, output: Path) -> None:
    effective_edit_ref_guidance_scale = (
        args.edit_ref_guidance_scale if args.guidance_mode in {"image_ref", "both"} else 0.0
    )
    protocol = {
        "experiment": "completion_guided_removal_dece_rf",
        "source_method": args.source_method,
        "output_method": args.output_method,
        "reference_method": args.reference_method,
        "tasks": split_items(args.tasks),
        "seeds": split_items(args.seeds),
        "completion_prior": {
            "type": "same_support_classical_inpaint",
            "method": args.completion_method,
            "radius": args.inpaint_radius,
            "mask_dilate": args.inpaint_dilate,
            "mask_blur": args.inpaint_blur,
            "mask_threshold": args.inpaint_threshold,
        },
        "fixed_guidance": {
            "guidance_mode": args.guidance_mode,
            "edit_ref_guidance_scale": effective_edit_ref_guidance_scale,
            "edit_ref_chroma_mode": args.edit_ref_chroma_mode,
            "edit_ref_luma_preserve_scale": args.edit_ref_luma_preserve_scale,
            "edit_ref_gradient_preserve_scale": args.edit_ref_gradient_preserve_scale,
            "edit_ref_smooth_kernel": args.edit_ref_smooth_kernel,
            "edit_text_guidance_scale": args.edit_text_guidance_scale,
            "edit_ref_structure_mode": args.edit_ref_structure_mode,
            "completion_clean_delta_scale": args.completion_clean_delta_scale,
            "completion_clean_delta_schedule_start": args.completion_clean_delta_schedule_start,
            "completion_clean_delta_schedule_stop": args.completion_clean_delta_schedule_stop,
            "completion_clean_delta_schedule_power": args.completion_clean_delta_schedule_power,
            "with_clean_fill": args.with_clean_fill,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run generic completion-guided removal experiments for DeCE-RF.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--tasks", default=" ".join(DEFAULT_TASKS))
    parser.add_argument("--seeds", default="10")
    parser.add_argument("--source-method", default="support_v3_controller_rmsgap")
    parser.add_argument("--output-method", default=DEFAULT_OUTPUT_METHOD)
    parser.add_argument("--reference-method", default="same_support_inpaint_telea")
    parser.add_argument("--completion-method", choices=("telea", "ns"), default="telea")
    parser.add_argument("--guidance-mode", choices=("image_ref", "clean_delta", "both"), default="image_ref")
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--inpaint-dilate", type=int, default=3)
    parser.add_argument("--inpaint-blur", type=float, default=0.0)
    parser.add_argument("--inpaint-threshold", type=int, default=45)
    parser.add_argument("--edit-ref-guidance-scale", type=float, default=0.55)
    parser.add_argument("--edit-ref-chroma-mode", choices=("yuv", "yuv_direction"), default="yuv")
    parser.add_argument("--edit-ref-luma-preserve-scale", type=float, default=0.80)
    parser.add_argument("--edit-ref-gradient-preserve-scale", type=float, default=0.40)
    parser.add_argument("--edit-ref-smooth-kernel", type=int, default=1)
    parser.add_argument("--edit-ref-structure-mode", choices=("source", "reference", "none"), default="source")
    parser.add_argument("--completion-clean-delta-scale", type=float, default=0.55)
    parser.add_argument("--completion-clean-delta-schedule-start", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-schedule-stop", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-schedule-power", type=float, default=1.0)
    parser.add_argument("--edit-text-guidance-scale", type=float, default=0.02)
    parser.add_argument("--with-clean-fill", action="store_true")
    parser.add_argument("--removal-fill-scale", type=float, default=0.70)
    parser.add_argument("--removal-suppression-scale", type=float, default=0.35)
    parser.add_argument("--removal-ring-rec-scale", type=float, default=0.40)
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=Path("experiments/support_v3_2026-05-11/removal_completion_ref_protocol.json"),
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.with_clean_fill and args.output_method == DEFAULT_OUTPUT_METHOD:
        args.output_method = f"{DEFAULT_OUTPUT_METHOD}_cleanfill"
    if args.guidance_mode == "clean_delta" and args.output_method == DEFAULT_OUTPUT_METHOD:
        args.output_method = "support_v3_controller_rmsgap_completion_clean_delta"

    if not args.dry_run:
        write_protocol(args, args.root / args.protocol_output)

    for seed in split_items(args.seeds):
        for task in split_items(args.tasks):
            ref_dir = args.root / "outputs" / "pretty_matrix" / task / args.reference_method / f"seed_{seed}"
            ensure_reference(args, task, seed, ref_dir)
            out_dir, env, cmd = build_task_command(args, task, seed)
            if args.skip_existing and (out_dir / "result.png").exists():
                print(f"skip existing: {out_dir}")
                continue
            command_line = shell_join(env, cmd)
            print(command_line)
            if not args.dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "command.txt").write_text(command_line + "\n", encoding="utf-8")
            if not args.dry_run and not args.prepare_only:
                subprocess.run(cmd, cwd=args.root, env={**os.environ, **env}, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
