#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from pathlib import Path

from run_removal_completion_ref_experiment import (
    DEFAULT_TASKS,
    command_set,
    ensure_reference,
    load_json,
    parse_recorded_command,
    shell_join,
    split_items,
)


DEFAULT_OUTPUT_METHOD = "support_v3_controller_rmsgap_completion_clean_delta_gated_highconf"


def load_reliability(path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[str(row["task"])] = float(row["R"])
    return rows


def gate_factor(score: float, low: float, high: float, medium_scale: float) -> float:
    if score >= high:
        return 1.0
    if medium_scale > 0.0 and score >= low:
        return medium_scale
    return 0.0


def build_command(args: argparse.Namespace, task: str, seed: str, score: float) -> tuple[Path, dict[str, str], list[str], dict[str, float | str]]:
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
        raise FileNotFoundError(f"Missing completion reference for {task} seed {seed}: {ref_dir}")

    env, cmd = parse_recorded_command(source_command)
    factor = gate_factor(score, args.gate_low, args.gate_high, args.gate_medium_scale)
    scale = args.completion_clean_delta_scale * factor

    env["TASK"] = task
    env["METHOD"] = args.output_method
    env["SEED"] = seed

    command_set(cmd, "--output", out_dir / "result.png")
    command_set(cmd, "--stats-output", out_dir / "stats.json")
    command_set(cmd, "--metadata-output", out_dir / "metadata.json")
    command_set(cmd, "--mask-output-dir", out_dir / "masks")

    if factor > 0.0:
        command_set(cmd, "--edit-text-guidance-scale", args.edit_text_guidance_scale)
        command_set(cmd, "--completion-clean-delta-scale", scale)
        command_set(cmd, "--completion-clean-delta-image", ref_image)
        command_set(cmd, "--completion-clean-delta-mask", ref_mask)
        command_set(cmd, "--completion-clean-delta-schedule-start", args.completion_clean_delta_schedule_start)
        command_set(cmd, "--completion-clean-delta-schedule-stop", args.completion_clean_delta_schedule_stop)
        command_set(cmd, "--completion-clean-delta-schedule-power", args.completion_clean_delta_schedule_power)
        command_set(cmd, "--removal-controller-mode", "none")

    gate = {
        "task": task,
        "seed": seed,
        "R": score,
        "gate_factor": factor,
        "completion_clean_delta_scale": scale,
        "mode": "clean_delta" if factor > 0.0 else "default_source",
    }
    return out_dir, env, cmd, gate


def write_protocol(args: argparse.Namespace, gates: list[dict[str, float | str]]) -> None:
    protocol = {
        "experiment": "reliability_gated_completion_clean_delta",
        "source_method": args.source_method,
        "output_method": args.output_method,
        "reference_method": args.reference_method,
        "reliability_csv": str(args.reliability_csv),
        "tasks": split_items(args.tasks),
        "seeds": split_items(args.seeds),
        "gate": {
            "type": "high_confidence" if args.gate_medium_scale <= 0.0 else "tiered",
            "low": args.gate_low,
            "high": args.gate_high,
            "medium_scale": args.gate_medium_scale,
            "base_completion_clean_delta_scale": args.completion_clean_delta_scale,
        },
        "fixed_guidance": {
            "edit_text_guidance_scale_when_enabled": args.edit_text_guidance_scale,
            "completion_clean_delta_schedule_start": args.completion_clean_delta_schedule_start,
            "completion_clean_delta_schedule_stop": args.completion_clean_delta_schedule_stop,
            "completion_clean_delta_schedule_power": args.completion_clean_delta_schedule_power,
        },
        "per_task_gate": gates,
    }
    args.protocol_output.parent.mkdir(parents=True, exist_ok=True)
    args.protocol_output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reliability-gated clean-delta removal experiments.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--tasks", default=" ".join(DEFAULT_TASKS))
    parser.add_argument("--seeds", default="10")
    parser.add_argument("--source-method", default="support_v3_controller_rmsgap")
    parser.add_argument("--output-method", default=DEFAULT_OUTPUT_METHOD)
    parser.add_argument("--reference-method", default="same_support_inpaint_telea")
    parser.add_argument("--completion-method", choices=("telea", "ns"), default="telea")
    parser.add_argument("--reliability-csv", type=Path, default=Path("experiments/support_v3_2026-05-11/prior_reliability/completion_prior_reliability_seed10.csv"))
    parser.add_argument("--gate-low", type=float, default=0.50)
    parser.add_argument("--gate-high", type=float, default=0.50)
    parser.add_argument("--gate-medium-scale", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-scale", type=float, default=0.55)
    parser.add_argument("--completion-clean-delta-schedule-start", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-schedule-stop", type=float, default=0.0)
    parser.add_argument("--completion-clean-delta-schedule-power", type=float, default=1.0)
    parser.add_argument("--edit-text-guidance-scale", type=float, default=0.02)
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--inpaint-dilate", type=int, default=3)
    parser.add_argument("--inpaint-blur", type=float, default=0.0)
    parser.add_argument("--inpaint-threshold", type=int, default=45)
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--protocol-output", type=Path, default=Path("experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_highconf_protocol.json"))
    args = parser.parse_args()

    args.root = args.root.resolve()
    if not args.reliability_csv.is_absolute():
        args.reliability_csv = args.root / args.reliability_csv
    if not args.protocol_output.is_absolute():
        args.protocol_output = args.root / args.protocol_output

    reliability = load_reliability(args.reliability_csv)
    gates: list[dict[str, float | str]] = []
    jobs: list[tuple[Path, dict[str, str], list[str]]] = []
    for seed in split_items(args.seeds):
        for task in split_items(args.tasks):
            if task not in reliability:
                raise KeyError(f"Missing reliability score for task: {task}")
            ref_dir = args.root / "outputs" / "pretty_matrix" / task / args.reference_method / f"seed_{seed}"
            ensure_reference(args, task, seed, ref_dir)
            out_dir, env, cmd, gate = build_command(args, task, seed, reliability[task])
            gates.append(gate)
            jobs.append((out_dir, env, cmd))

    if not args.dry_run:
        write_protocol(args, gates)

    for gate in gates:
        print(
            f"[gate] {gate['task']} seed={gate['seed']} "
            f"R={float(gate['R']):.3f} factor={float(gate['gate_factor']):.2f} "
            f"scale={float(gate['completion_clean_delta_scale']):.3f} mode={gate['mode']}"
        )

    for out_dir, env, cmd in jobs:
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
