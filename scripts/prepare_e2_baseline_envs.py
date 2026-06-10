#!/usr/bin/env python3
"""Create one isolated Python environment per E2 baseline.

The goal is not to certify paper comparability. It is to make every downloaded
baseline reach a separately logged environment state so command smoke and later
Core-6 adapters do not pollute the DeCE-RF runtime.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvSpec:
    repo_name: str
    env_name: str
    python: str
    install: tuple[tuple[str, ...], ...]
    note: str = ""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = Path(os.environ.get("BASELINE_ROOT", PROJECT_ROOT / "_baselines"))


def make_specs(src: Path) -> tuple[EnvSpec, ...]:
    return (
    EnvSpec(
        "FireFlow",
        "fireflow-py310",
        "/usr/bin/python3.10",
        (
            (
                "-e",
                str(src / "FireFlow"),
                "transformers==4.46.3",
                "tokenizers==0.20.3",
                "huggingface-hub==0.26.5",
            ),
        ),
        "Project pyproject install with HF pins verified against torch 2.4/cu121.",
    ),
    EnvSpec(
        "FlowAlign",
        "flowalign-py310",
        "/usr/bin/python3.10",
        (("-r", str(src / "FlowAlign/requirements.txt")), ("torch", "torchvision", "pillow", "opencv-python-headless")),
        "Requirements plus torch/image IO.",
    ),
    EnvSpec(
        "FlowEdit",
        "flowedit-py310",
        "/usr/bin/python3.10",
        (
            (
                "torch",
                "torchvision",
                "diffusers==0.33.1",
                "transformers==4.47.1",
                "accelerate==1.2.1",
                "tokenizers==0.21.0",
                "sentencepiece==0.2.0",
                "protobuf==3.20.3",
                "pyyaml",
                "pillow",
                "numpy==1.26.0",
                "tqdm",
                "safetensors",
            ),
        ),
        "No requirements file; dependency set inferred from imports and FlowAlign pins.",
    ),
    EnvSpec(
        "SplitFlow",
        "splitflow-py310",
        "/usr/bin/python3.10",
        (
            (
                "torch",
                "torchvision",
                "diffusers==0.33.1",
                "transformers==4.47.1",
                "accelerate==1.2.1",
                "tokenizers==0.21.0",
                "sentencepiece==0.2.0",
                "protobuf==3.20.3",
                "pyyaml",
                "pillow",
                "numpy==1.26.0",
                "tqdm",
                "matplotlib",
                "safetensors",
            ),
        ),
        "No requirements file; dependency set inferred from imports and FlowAlign pins.",
    ),
    EnvSpec(
        "RF-Solver-Edit",
        "rf-solver-edit-py310",
        "/usr/bin/python3.10",
        (
            (
                "-e",
                str(src / "RF-Solver-Edit/FLUX_Image_Edit"),
                "transformers==4.46.3",
                "tokenizers==0.20.3",
                "huggingface-hub==0.26.5",
            ),
        ),
        "FLUX image-edit subproject install with HF pins; video subproject is out of E2 image scope.",
    ),
    EnvSpec(
        "ReFlex",
        "reflex-py310",
        "/usr/bin/python3.10",
        (("-r", str(src / "ReFlex/requirements.txt")), ("torchao==0.5.0",)),
        "Requirements file install with torchao pinned for torch 2.4 compatibility.",
    ),
    EnvSpec(
        "stable-flow",
        "stable-flow-py311",
        "/usr/bin/python3.11",
        (
            ("-e", str(src / "stable-flow")),
            ("torch", "torchvision", "accelerate", "transformers==4.44.2", "sentencepiece", "protobuf", "pillow", "numpy"),
        ),
        "Local diffusers-style package install plus runtime deps.",
    ),
    EnvSpec(
        "ZONE",
        "zone-py310",
        "/usr/bin/python3.10",
        (
            (
                "absl-py==2.0.0",
                "accelerate==0.23.0",
                "albumentations==0.5.2",
                "diffusers==0.18.0",
                "einops==0.6.1",
                "transformers==4.27.4",
                "huggingface-hub==0.17.3",
                "tokenizers==0.13.3",
                "gradio==3.45.0",
                "opencv-python-headless",
                "pillow",
                "numpy",
                "scipy",
                "scikit-image",
                "matplotlib",
                "tqdm",
                "ftfy",
                "torch",
                "torchvision",
                "segment-anything",
            ),
        ),
        "Sanitized from requirements; skips unavailable py38 detectron2 wheel and ssh CLIP URL.",
    ),
    EnvSpec(
        "instruct-pix2pix",
        "instruct-pix2pix-py39",
        "/usr/bin/python3.9",
        (
            (
                "torch",
                "torchvision",
                "diffusers",
                # Python 3.9 is not available on GPU01, so this environment
                # falls back to Python 3.11. The original transformers 4.19
                # pin pulls tokenizers without cp311 wheels and needs Rust.
                "transformers==4.30.2",
                "omegaconf==2.1.1",
                "pytorch-lightning==1.4.2",
                "einops==0.3.0",
                "opencv-python-headless",
                "pillow",
                "numpy==1.23.5",
                "kornia==0.6",
                "torchmetrics==0.6.0",
                "datasets==2.8.0",
                "gradio",
                "seaborn",
                "invisible-watermark",
                "k-diffusion",
            ),
        ),
        "Conda env requested Python 3.8/torch 1.11; py39 is closest installed interpreter.",
    ),
    EnvSpec(
        "pix2pix-zero",
        "pix2pix-zero-py310",
        "/usr/bin/python3.10",
        (
            (
                "accelerate==0.20.3",
                "diffusers==0.13.1",
                "einops",
                "gradio",
                "ipython",
                "numpy",
                "opencv-python-headless",
                "pillow",
                "psutil",
                "tqdm",
                "transformers==4.29.2",
                "huggingface-hub==0.25.2",
                "tokenizers==0.13.3",
                "openai",
                "torch",
                "torchvision",
            ),
        ),
        "Pip subset from environment.yml; omits lavis until needed for generation.",
    ),
    EnvSpec(
        "MasaCtrl",
        "masactrl-py310",
        "/usr/bin/python3.10",
        (
            ("-r", str(src / "MasaCtrl/requirements.txt")),
            (
                "torch",
                "torchvision",
                "pillow",
                "numpy",
                "transformers==4.29.2",
                "accelerate==0.20.3",
                "huggingface-hub==0.17.3",
                "tokenizers==0.13.3",
            ),
        ),
        "Requirements file plus torch/image IO and HF pins verified by smoke tests.",
    ),
    EnvSpec(
        "prompt-to-prompt",
        "prompt-to-prompt-py310",
        "/usr/bin/python3.10",
        (("-r", str(src / "prompt-to-prompt/requirements.txt")), ("torch", "torchvision", "pillow", "numpy")),
        "Requirements file plus torch/image IO.",
    ),
    EnvSpec(
        "h-edit",
        "h-edit-py310",
        "/usr/bin/python3.10",
        (
            (
                "torch",
                "torchvision",
                "diffusers==0.15.0",
                "transformers==4.29.2",
                "huggingface-hub==0.17.3",
                "accelerate==0.20.3",
                "tokenizers==0.13.3",
                "ftfy",
                "opencv-python-headless",
                "pillow",
                "numpy",
                "scipy",
                "tqdm",
                "matplotlib",
                "einops",
                "omegaconf",
                "pytorch-lightning",
                "nltk==3.9.1",
            ),
        ),
        "No requirements file; dependency set inferred from text-guided imports and smoke-tested pins.",
    ),
    EnvSpec(
        "ledits_pp",
        "ledits-pp-py310",
        "/usr/bin/python3.10",
        (
            ("--no-build-isolation", "-e", str(src / "ledits_pp")),
            (
                "diffusers==0.20.2",
                "transformers==4.33.3",
                "accelerate==0.23.0",
                "huggingface-hub==0.17.3",
                "tokenizers==0.13.3",
            ),
        ),
        "Project setup install with HF pins verified by import smoke test.",
    ),
    EnvSpec(
        "OT-RF",
        "ot-rf-otip-py310",
        "/usr/bin/python3.10",
        (
            (
                "torch==2.4.1",
                "torchvision==0.19.1",
                "torchaudio==2.4.1",
                "diffusers==0.30.1",
                "transformers==4.46.3",
                "accelerate==1.0.1",
                "tokenizers==0.20.3",
                "huggingface-hub==0.29.1",
                "safetensors==0.5.3",
                "sentencepiece==0.2.0",
                "protobuf==5.29.3",
                "pyyaml==6.0.2",
                "pillow==9.5.0",
                "numpy==1.24.4",
                "scipy==1.10.1",
                "scikit-learn==1.3.2",
                "matplotlib==3.7.5",
                "tqdm==4.67.1",
            ),
        ),
        "OT-RF/OTIP at abca084; FLUX/SD3 FlowEdit-style entrypoint.",
    ),
    EnvSpec(
        "DeltaRectifiedFlowSampling",
        "dvrf-py310",
        "/usr/bin/python3.10",
        (
            (
                "torch==2.4.1",
                "torchvision==0.19.1",
                "torchaudio==2.4.1",
                "diffusers==0.30.1",
                "transformers==4.46.3",
                "accelerate==1.0.1",
                "tokenizers==0.20.3",
                "huggingface-hub==0.29.1",
                "safetensors==0.5.3",
                "sentencepiece==0.2.0",
                "protobuf==5.29.3",
                "pyyaml==6.0.2",
                "pillow==9.5.0",
                "numpy==1.24.4",
                "scipy==1.10.1",
                "scikit-learn==1.3.2",
                "matplotlib==3.7.5",
                "tqdm==4.67.1",
                "gpustat==1.1.1",
                "torchmetrics==1.5.2",
            ),
        ),
        "Delta Rectified Flow Sampling at 567b28b; pip subset from drfs_environment.yml.",
    ),
    )


def run_cmd(cmd: list[str], log_path: Path, cwd: Path | None = None, timeout: int = 1800) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout}s\n")
            return 124, "timeout"
    return proc.returncode, "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"


def resolve_python(requested: str) -> tuple[str, str]:
    if Path(requested).exists() or shutil.which(requested):
        return requested, "requested"
    for fallback in ("/usr/bin/python3.11", "/usr/bin/python3.12", sys.executable, "python3"):
        if fallback and (Path(fallback).exists() or shutil.which(fallback)):
            return fallback, f"fallback_from_{requested}"
    return requested, "missing"


def env_python(env_dir: Path) -> Path:
    return env_dir / "bin/python"


def create_env_cmd(python: str, env_dir: Path) -> list[str]:
    if shutil.which("uv"):
        return ["uv", "venv", "--python", python, str(env_dir)]
    return [python, "-m", "venv", str(env_dir)]


def pip_install_cmd(python: Path, packages: tuple[str, ...]) -> list[str]:
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", str(python), *packages]
    return [str(python), "-m", "pip", "install", *packages]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default=str(BASELINE_ROOT))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    env_root = baseline_root / "envs"
    log_root = baseline_root / "logs/env_install"
    registry_path = baseline_root / "e2_baseline_env_registry.csv"
    env_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.only or [])
    rows: list[dict[str, str]] = []

    for spec in make_specs(baseline_root / "src"):
        if selected and spec.repo_name not in selected and spec.env_name not in selected:
            continue

        requested_python = spec.python
        actual_python, python_status = resolve_python(requested_python)
        env_dir = env_root / spec.env_name
        log_path = log_root / f"{spec.env_name}.log"
        if log_path.exists():
            log_path.unlink()

        create_status = "skipped_existing"
        if args.recreate and env_dir.exists():
            code, create_status = run_cmd(["rm", "-rf", str(env_dir)], log_path, timeout=120)
            if code != 0:
                rows.append(
                    {
                        "repo_name": spec.repo_name,
                        "env_name": spec.env_name,
                        "env_path": str(env_dir),
                        "python": actual_python,
                        "requested_python": requested_python,
                        "python_status": python_status,
                        "env_python": str(env_python(env_dir)),
                        "create_status": create_status,
                        "install_status": "not_run",
                        "log_path": str(log_path),
                        "note": spec.note,
                    }
                )
                continue

        if not env_python(env_dir).exists():
            code, create_status = run_cmd(
                create_env_cmd(actual_python, env_dir),
                log_path,
                timeout=300,
            )
        if not env_python(env_dir).exists():
            rows.append(
                {
                    "repo_name": spec.repo_name,
                    "env_name": spec.env_name,
                    "env_path": str(env_dir),
                    "python": actual_python,
                    "requested_python": requested_python,
                    "python_status": python_status,
                    "env_python": str(env_python(env_dir)),
                    "create_status": create_status,
                    "install_status": "env_create_failed",
                    "log_path": str(log_path),
                    "note": spec.note,
                }
            )
            continue

        install_status = "ok"
        bootstrap_cmd = pip_install_cmd(env_python(env_dir), ("-U", "pip", "setuptools<81", "wheel"))
        code, status = run_cmd(bootstrap_cmd, log_path, timeout=600)
        if code != 0:
            install_status = f"bootstrap_{status}"
        else:
            for packages in spec.install:
                cmd = pip_install_cmd(env_python(env_dir), packages)
                code, status = run_cmd(cmd, log_path, timeout=args.timeout_s)
                if code != 0:
                    install_status = status
                    break

        rows.append(
            {
                "repo_name": spec.repo_name,
                "env_name": spec.env_name,
                "env_path": str(env_dir),
                "python": actual_python,
                "requested_python": requested_python,
                "python_status": python_status,
                "env_python": str(env_python(env_dir)),
                "create_status": create_status,
                "install_status": install_status,
                "log_path": str(log_path),
                "note": spec.note,
            }
        )

    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            row["repo_name"],
            row["env_name"],
            row["create_status"],
            row["install_status"],
            row["log_path"],
        )
    print(f"Wrote {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
