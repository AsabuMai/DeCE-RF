#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J pin-e2-hf
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/env_repair_20260603/hf_pin_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/env_repair_20260603/hf_pin_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
LOGDIR="$PROJECT/_baselines/logs/env_repair_20260603"
PIP_CACHE_DIR="$PROJECT/.pip_cache"
export PIP_CACHE_DIR PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1

cd "$PROJECT"
mkdir -p "$LOGDIR" "$PIP_CACHE_DIR"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to pin envs outside a100-01" >&2
  exit 2
fi

pin_install() {
  local envname="$1"
  shift
  local py="_baselines/envs/$envname/bin/python"
  echo "==== HF/version repair: $envname :: $*"
  "$py" -m pip install --ignore-installed --no-deps "$@" > "$LOGDIR/${envname}_hf_pins.log" 2>&1
  "$py" - "$envname" <<'PY'
import re
import shutil
import site
import sys
from pathlib import Path

targets = {
    "zone-py310": {
        "diffusers": "0.18.0",
        "transformers": "4.27.4",
        "accelerate": "0.23.0",
        "huggingface_hub": "0.17.3",
        "tokenizers": "0.13.3",
    },
    "masactrl-py310": {
        "diffusers": "0.15.0",
        "transformers": "4.29.2",
        "accelerate": "0.20.3",
        "huggingface_hub": "0.17.3",
        "tokenizers": "0.13.3",
    },
    "h-edit-py310": {
        "diffusers": "0.15.0",
        "transformers": "4.29.2",
        "accelerate": "0.20.3",
        "huggingface_hub": "0.17.3",
        "tokenizers": "0.13.3",
    },
    "ledits-pp-py310": {
        "diffusers": "0.20.2",
        "transformers": "4.33.3",
        "accelerate": "0.23.0",
        "huggingface_hub": "0.17.3",
        "tokenizers": "0.13.3",
    },
    "pix2pix-zero-py310": {
        "diffusers": "0.15.0",
        "transformers": "4.29.2",
        "accelerate": "0.20.3",
        "huggingface_hub": "0.17.3",
        "tokenizers": "0.13.3",
    },
}

envname = sys.argv[1]
desired = targets[envname]
site_dirs = [Path(p) for p in site.getsitepackages()]
for site_dir in site_dirs:
    if not site_dir.exists():
        continue
    for pkg, version in desired.items():
        prefixes = {pkg, pkg.replace("_", "-")}
        for dist in site_dir.glob("*.dist-info"):
            name = dist.name[:-10]
            for prefix in prefixes:
                match = re.fullmatch(re.escape(prefix) + r"-(.+)", name, flags=re.I)
                if match and match.group(1) != version:
                    shutil.rmtree(dist)
                    print(f"removed stale dist-info {dist.name}")
PY
  "$py" - <<'PY'
import importlib.metadata as m
for p in ["diffusers", "transformers", "accelerate", "huggingface-hub", "tokenizers"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "MISSING")
PY
}

pin_install zone-py310 \
  diffusers==0.18.0 transformers==4.27.4 accelerate==0.23.0 huggingface-hub==0.17.3 tokenizers==0.13.3
pin_install masactrl-py310 \
  diffusers==0.15.0 transformers==4.29.2 accelerate==0.20.3 huggingface-hub==0.17.3 tokenizers==0.13.3
pin_install h-edit-py310 \
  diffusers==0.15.0 transformers==4.29.2 accelerate==0.20.3 huggingface-hub==0.17.3 tokenizers==0.13.3
pin_install ledits-pp-py310 \
  diffusers==0.20.2 transformers==4.33.3 accelerate==0.23.0 huggingface-hub==0.17.3 tokenizers==0.13.3
pin_install pix2pix-zero-py310 \
  diffusers==0.15.0 transformers==4.29.2 accelerate==0.20.3 huggingface-hub==0.17.3 tokenizers==0.13.3

echo "hf_pins_complete"
