#!/usr/bin/env python3
"""Smoke-test rebuilt baseline entrypoints without running model generation."""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "_baselines"
SRC = BASE / "src"
REGISTRY = BASE / "e2_baseline_env_registry.csv"
OUT = BASE / "e2_baseline_smoke_results.csv"


@dataclass(frozen=True)
class SmokeSpec:
    repo_name: str
    kind: str
    cwd: Path
    args: tuple[str, ...]
    note: str = ""


SPECS = {
    "FireFlow": SmokeSpec(
        "FireFlow", "cmd_help", SRC / "FireFlow", ("src/edit.py", "--help")
    ),
    "FlowAlign": SmokeSpec(
        "FlowAlign", "cmd_help", SRC / "FlowAlign", ("run_edit.py", "--help")
    ),
    "FlowEdit": SmokeSpec(
        "FlowEdit", "cmd_help", SRC / "FlowEdit", ("run_script.py", "--help")
    ),
    "SplitFlow": SmokeSpec(
        "SplitFlow", "cmd_help", SRC / "SplitFlow", ("run_script.py", "--help")
    ),
    "RF-Solver-Edit": SmokeSpec(
        "RF-Solver-Edit",
        "cmd_help",
        SRC / "RF-Solver-Edit/FLUX_Image_Edit",
        ("src/edit.py", "--help"),
    ),
    "ReFlex": SmokeSpec(
        "ReFlex", "cmd_help", SRC / "ReFlex", ("img_edit.py", "--help")
    ),
    "stable-flow": SmokeSpec(
        "stable-flow",
        "cmd_help",
        SRC / "stable-flow",
        ("run_stable_flow.py", "--help"),
    ),
    "ZONE": SmokeSpec(
        "ZONE",
        "import_deps",
        SRC / "ZONE",
        ("import torch, diffusers, transformers, accelerate, PIL, cv2, numpy",),
        "Avoid importing inference.py because it starts downloading model weights at import time.",
    ),
    "instruct-pix2pix": SmokeSpec(
        "instruct-pix2pix",
        "cmd_help",
        SRC / "instruct-pix2pix",
        ("edit_cli.py", "--help"),
    ),
    "pix2pix-zero": SmokeSpec(
        "pix2pix-zero",
        "cmd_help",
        SRC / "pix2pix-zero",
        ("src/edit_real.py", "--help"),
        "Use the experiment CLI rather than app_gradio; the UI has a separate gradio/hub version contract.",
    ),
    "MasaCtrl": SmokeSpec(
        "MasaCtrl",
        "import_deps",
        SRC / "MasaCtrl",
        (
            "import torch, numpy, tqdm, einops, omegaconf, diffusers, torchvision, pytorch_lightning",
        ),
        "Avoid importing run_synthesis_sdxl.py because it loads SDXL at module import.",
    ),
    "prompt-to-prompt": SmokeSpec(
        "prompt-to-prompt",
        "import",
        SRC / "prompt-to-prompt",
        ("import ptp_utils, seq_aligner",),
    ),
    "h-edit": SmokeSpec(
        "h-edit",
        "cmd_help",
        SRC / "h-edit/text-guided",
        ("main_demo.py", "--help"),
    ),
    "ledits_pp": SmokeSpec(
        "ledits_pp", "import", SRC / "ledits_pp", ("import leditspp",)
    ),
    "OT-RF": SmokeSpec(
        "OT-RF",
        "cmd_help",
        SRC / "OT-RF",
        ("run_script.py", "--help"),
    ),
    "DeltaRectifiedFlowSampling": SmokeSpec(
        "DeltaRectifiedFlowSampling",
        "cmd_help",
        SRC / "DeltaRectifiedFlowSampling",
        ("edit.py", "--help"),
    ),
}


def truncate(text: str, limit: int = 2400) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    text = text.replace("\r", "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def missing_modules(stderr: str) -> str:
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    names = re.findall(r"ModuleNotFoundError: No module named '([^']+)'", stderr)
    names += re.findall(r"ImportError: No module named ([^\s]+)", stderr)
    return ";".join(sorted(set(names)))


def run_one(row: dict[str, str]) -> dict[str, str]:
    spec = SPECS[row["repo_name"]]
    env = os.environ.copy()
    py_path = str(spec.cwd)
    if spec.repo_name in {"FireFlow", "RF-Solver-Edit"}:
        py_path = str(spec.cwd / "src")
    env["PYTHONPATH"] = py_path + os.pathsep + env.get("PYTHONPATH", "")

    if spec.kind == "cmd_help":
        cmd = [row["env_python"], *spec.args]
    elif spec.kind in {"import", "import_deps"}:
        cmd = [row["env_python"], "-c", spec.args[0]]
    else:
        raise ValueError(spec.kind)

    try:
        proc = subprocess.run(
            cmd,
            cwd=spec.cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
        )
        status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

    return {
        "repo_name": row["repo_name"],
        "env_name": row["env_name"],
        "kind": spec.kind,
        "status": status,
        "missing_modules": missing_modules(stderr),
        "command": " ".join(str(part) for part in cmd),
        "cwd": str(spec.cwd),
        "note": spec.note,
        "stdout_tail": truncate(stdout),
        "stderr_tail": truncate(stderr),
    }


def main() -> int:
    rows = list(csv.DictReader(REGISTRY.open(newline="", encoding="utf-8")))
    results = [run_one(row) for row in rows]
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    for result in results:
        print(
            f"{result['repo_name']:<18s} {result['kind']:<11s} "
            f"{result['status']:<8s} missing={result['missing_modules']}"
        )
    return 1 if any(r["status"] != "ok" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
