#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J repair-smoke
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/env_repair_20260603/smoke_repair_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/env_repair_20260603/smoke_repair_%j.err

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
  echo "Refusing to repair envs outside a100-01" >&2
  exit 2
fi

clean_pkgs() {
  local envname="$1"
  shift
  local py="_baselines/envs/$envname/bin/python"
  "$py" - "$@" <<'PY'
import shutil
import site
import sys
from pathlib import Path

names = sys.argv[1:]
site_dirs = [Path(p) for p in site.getsitepackages()]
for site_dir in site_dirs:
    if not site_dir.exists():
        continue
    for name in names:
        candidates = {
            name,
            name.replace("-", "_"),
            name.replace("_", "-"),
        }
        for candidate in candidates:
            for path in site_dir.glob(candidate + "*"):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"removed {path}")
                elif path.exists():
                    path.unlink()
                    print(f"removed {path}")
PY
}

show_versions() {
  local envname="$1"
  local py="_baselines/envs/$envname/bin/python"
  "$py" - <<'PY'
import importlib.metadata as m
for p in ["torch", "transformers", "tokenizers", "huggingface-hub", "diffusers", "accelerate", "gradio", "nltk"]:
    try:
        print(p, m.version(p))
    except Exception:
        pass
PY
}

repair_flux_env() {
  local envname="$1"
  local py="_baselines/envs/$envname/bin/python"
  echo "==== repair flux-style env: $envname"
  clean_pkgs "$envname" transformers tokenizers tokenizers.libs huggingface_hub huggingface-hub
  "$py" -m pip install --no-deps \
    transformers==4.46.3 tokenizers==0.20.3 huggingface-hub==0.26.5 \
    > "$LOGDIR/${envname}_smoke_repair.log" 2>&1
  show_versions "$envname"
}

repair_pix2pix_zero() {
  local envname=pix2pix-zero-py310
  local py="_baselines/envs/$envname/bin/python"
  echo "==== repair pix2pix-zero clean HF stack"
  clean_pkgs "$envname" diffusers transformers accelerate tokenizers tokenizers.libs huggingface_hub huggingface-hub
  "$py" -m pip install --no-deps \
    diffusers==0.13.1 transformers==4.29.2 accelerate==0.20.3 huggingface-hub==0.25.2 tokenizers==0.13.3 \
    > "$LOGDIR/${envname}_smoke_repair.log" 2>&1
  show_versions "$envname"
}

repair_hedit() {
  local envname=h-edit-py310
  local py="_baselines/envs/$envname/bin/python"
  echo "==== repair h-edit nltk"
  "$py" -m pip install "nltk==3.9.1" > "$LOGDIR/${envname}_nltk.log" 2>&1
  show_versions "$envname"
}

repair_flux_env fireflow-py310
repair_flux_env rf-solver-edit-py310
repair_pix2pix_zero
repair_hedit

echo smoke_failure_repairs_complete
