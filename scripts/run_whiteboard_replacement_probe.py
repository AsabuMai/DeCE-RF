#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


SOURCE_TASK = "whiteboard_remove_yellow_letter"
SOURCE_METHOD = "support_v3_controller_rmsgap"


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    target_phrase: str
    prompt: str
    operation: str
    new_tokens: str


VARIANTS = [
    Variant(
        name="whiteboard_probe_blank",
        label="blank",
        target_phrase="blank whiteboard surface",
        prompt=(
            "A close-up photo of the same whiteboard with the yellow letter I removed, "
            "leaving blank whiteboard surface in that spot, preserving faint marker scribbles "
            "and all other magnetic letters."
        ),
        operation="remove_object",
        new_tokens="",
    ),
    Variant(
        name="whiteboard_probe_blue_letter_t",
        label="blue letter T",
        target_phrase="blue magnetic plastic letter T",
        prompt=(
            "A close-up photo of the same whiteboard with the yellow letter I replaced by "
            "a blue magnetic plastic letter T in the same spot, preserving all other letters, "
            "whiteboard surface, marker scribbles, lighting, and composition."
        ),
        operation="replace",
        new_tokens="blue,letter,T",
    ),
    Variant(
        name="whiteboard_probe_red_letter_a",
        label="red letter A",
        target_phrase="red magnetic plastic letter A",
        prompt=(
            "A close-up photo of the same whiteboard with the yellow letter I replaced by "
            "a red magnetic plastic letter A in the same spot, preserving all other letters, "
            "whiteboard surface, marker scribbles, lighting, and composition."
        ),
        operation="replace",
        new_tokens="red,letter,A",
    ),
    Variant(
        name="whiteboard_probe_blue_round_magnet",
        label="blue round magnet",
        target_phrase="blue round magnet",
        prompt=(
            "A close-up photo of the same whiteboard with the yellow letter I replaced by "
            "a blue round magnet in the same spot, preserving all other letters, whiteboard "
            "surface, marker scribbles, lighting, and composition."
        ),
        operation="replace",
        new_tokens="blue,round,magnet",
    ),
    Variant(
        name="whiteboard_probe_red_star_sticker",
        label="red star sticker",
        target_phrase="red star sticker",
        prompt=(
            "A close-up photo of the same whiteboard with the yellow letter I replaced by "
            "a red star sticker in the same spot, preserving all other letters, whiteboard "
            "surface, marker scribbles, lighting, and composition."
        ),
        operation="replace",
        new_tokens="red,star,sticker",
    ),
]

METHODS = {
    "support_v3_controller_rmsgap": "default",
    "support_v3_controller_rmsgap_replace_editor_v0": "replace_editor_v0",
    "support_v3_controller_rmsgap_replace_editor_v1": "replace_editor_v1",
}


def replace_option(cmd: list[str], option: str, value: str) -> None:
    if option not in cmd:
        cmd.extend([option, value])
        return
    idx = cmd.index(option)
    if idx + 1 >= len(cmd):
        raise ValueError(f"option {option} has no value")
    cmd[idx + 1] = value


def remove_option(cmd: list[str], option: str, takes_value: bool = True) -> None:
    while option in cmd:
        idx = cmd.index(option)
        del cmd[idx : idx + (2 if takes_value else 1)]


def source_command(root: Path, seed: int) -> tuple[dict[str, str], list[str]]:
    command_path = (
        root
        / "outputs"
        / "pretty_matrix"
        / SOURCE_TASK
        / SOURCE_METHOD
        / f"seed_{seed}"
        / "command.txt"
    )
    parts = shlex.split(command_path.read_text().strip())
    env = {}
    idx = 0
    while idx < len(parts) and "=" in parts[idx] and not parts[idx].startswith("--"):
        key, value = parts[idx].split("=", 1)
        env[key] = value
        idx += 1
    return env, parts[idx:]


def build_command(root: Path, seed: int, variant: Variant, method: str) -> tuple[Path, dict[str, str], list[str]]:
    src_env, cmd = source_command(root, seed)
    cmd = list(cmd)
    out_dir = root / "outputs" / "pretty_matrix" / variant.name / method / f"seed_{seed}"
    support_mask = (
        root
        / "outputs"
        / "pretty_matrix"
        / SOURCE_TASK
        / SOURCE_METHOD
        / f"seed_{seed}"
        / "masks"
        / "semantic_support.png"
    )

    replace_option(cmd, "--prompt", variant.prompt)
    replace_option(cmd, "--output", str(out_dir / "result.png"))
    replace_option(cmd, "--stats-output", str(out_dir / "stats.json"))
    replace_option(cmd, "--metadata-output", str(out_dir / "metadata.json"))
    replace_option(cmd, "--mask-output-dir", str(out_dir / "masks"))
    replace_option(cmd, "--support-mask", str(support_mask))
    replace_option(cmd, "--edit-operation", variant.operation)
    replace_option(cmd, "--attention-mask-target-words", variant.new_tokens or "yellow,letter,I")
    replace_option(cmd, "--host-tokens", "whiteboard,whiteboard surface")
    replace_option(cmd, "--removed-tokens", "yellow letter I,magnetic letter I,letter I")

    remove_option(cmd, "--new-tokens")
    if variant.new_tokens:
        cmd.extend(["--new-tokens", variant.new_tokens])

    # Keep this probe focused on target formation / replacement controllability.
    for opt in [
        "--completion-clean-delta-scale",
        "--completion-clean-delta-image",
        "--completion-clean-delta-mask",
        "--completion-clean-delta-schedule-start",
        "--completion-clean-delta-schedule-stop",
        "--completion-clean-delta-schedule-power",
    ]:
        remove_option(cmd, opt)

    # Reset replacement-editor-only flags before applying the selected method.
    for opt in [
        "--edit-text-source-prompt",
        "--edit-text-target-prompt",
        "--edit-local-target-prompt",
        "--edit-local-target-guidance-scale",
        "--edit-local-target-cfg-scale",
    ]:
        remove_option(cmd, opt)

    if variant.operation == "replace" and METHODS[method] in {"replace_editor_v0", "replace_editor_v1"}:
        replace_option(cmd, "--edit-hedit-guidance-scale", "0.88")
        replace_option(cmd, "--edit-text-guidance-scale", "0.22")
        replace_option(cmd, "--edit-text-source-scale", "1.15")
        replace_option(cmd, "--edit-text-core-weight", "1.45")
        replace_option(cmd, "--edit-text-subject-weight", "0.20")
        replace_option(cmd, "--edit-guidance-scale", "0.04")
        replace_option(cmd, "--edit-region-guidance-scale", "0.08")
        replace_option(cmd, "--edit-target-guidance-scale", "0.04")
        replace_option(cmd, "--edit-source-guidance-scale", "0.10")
        replace_option(cmd, "--rec-guidance-scale", "0.12")
        replace_option(cmd, "--trajectory-preserve-scale", "0.04")
        replace_option(cmd, "--region-target-transport-scale", "0.85")
        replace_option(cmd, "--region-target-outside-lock-scale", "0.80")
        replace_option(cmd, "--edit-initial-noise-scale", "0.18")
        replace_option(cmd, "--edit-initial-noise-region", "core")
        cmd.extend(
            [
                "--edit-text-source-prompt",
                "yellow magnetic plastic letter I",
                "--edit-text-target-prompt",
                variant.target_phrase,
            ]
        )
        if METHODS[method] == "replace_editor_v1":
            cmd.extend(
                [
                    "--edit-local-target-prompt",
                    f"A close-up photo of a {variant.target_phrase}.",
                    "--edit-local-target-guidance-scale",
                    "0.30",
                    "--edit-local-target-cfg-scale",
                    "8.0",
                ]
            )
    elif variant.operation == "replace":
        replace_option(cmd, "--edit-text-guidance-scale", "0.08")
        replace_option(cmd, "--edit-hedit-guidance-scale", "0.65")
        replace_option(cmd, "--rec-guidance-scale", "0.22")
        replace_option(cmd, "--trajectory-preserve-scale", "0.12")

    env = {**os.environ, **src_env}
    return out_dir, env, cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seeds", default="10 11 12")
    parser.add_argument("--variants", default=" ".join(variant.name for variant in VARIANTS))
    parser.add_argument("--methods", default=" ".join(METHODS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--protocol-output",
        default="experiments/support_v3_2026-05-11/whiteboard_replacement_probe_protocol.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    seeds = [int(x) for x in args.seeds.split()]
    variant_names = set(args.variants.replace(",", " ").split())
    method_names = set(args.methods.replace(",", " ").split())
    protocol = []
    for seed in seeds:
        for variant in VARIANTS:
            if variant.name not in variant_names:
                continue
            for method in METHODS:
                if method not in method_names:
                    continue
                out_dir, env, cmd = build_command(root, seed, variant, method)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "masks").mkdir(exist_ok=True)
                (out_dir / "command.txt").write_text(
                    " ".join(shlex.quote(x) for x in cmd) + "\n"
                )
                protocol.append(
                    {
                        "seed": seed,
                        "variant": variant.name,
                        "label": variant.label,
                        "target_phrase": variant.target_phrase,
                        "operation": variant.operation,
                        "method": method,
                        "output_dir": str(out_dir),
                        "support_mask": str(
                            root
                            / "outputs"
                            / "pretty_matrix"
                            / SOURCE_TASK
                            / SOURCE_METHOD
                            / f"seed_{seed}"
                            / "masks"
                            / "semantic_support.png"
                        ),
                    }
                )
                complete = all(
                    (out_dir / name).is_file()
                    for name in ["result.png", "stats.json", "metadata.json", "command.txt"]
                )
                print(f"[probe] {variant.name} {method} seed={seed}")
                if args.skip_existing and complete:
                    print(f"[probe] skip existing: {out_dir}")
                    continue
                if not args.dry_run:
                    subprocess.run(cmd, cwd=root, env=env, check=True)

    protocol_path = root / args.protocol_output
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(protocol_path)


if __name__ == "__main__":
    main()
