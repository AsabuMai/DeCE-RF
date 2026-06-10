#!/usr/bin/env python3
"""Lightweight validation for rebuilt E2 baseline environments."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path


CHECK_CODE = (
    "import importlib.util, sys; "
    "mods=['torch','diffusers','transformers','accelerate']; "
    "print(sys.version.split()[0], "
    "*['{}={}'.format(m, importlib.util.find_spec(m) is not None) for m in mods])"
)


def main() -> int:
    registry = Path("_baselines/e2_baseline_env_registry.csv")
    rows = list(csv.DictReader(registry.open(newline="", encoding="utf-8")))
    failures = 0
    for row in rows:
        proc = subprocess.run(
            [row["env_python"], "-c", CHECK_CODE],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
        print(f"{row['env_name']:<24s} {status:<8s} {proc.stdout.strip()}")
        if proc.returncode != 0:
            failures += 1
            tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
            if tail:
                print(tail)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
