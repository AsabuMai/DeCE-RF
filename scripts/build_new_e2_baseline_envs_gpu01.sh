#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:30:00
#SBATCH -J build-e2-new
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/build_new_envs_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/build_new_envs_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
ENV_ROOT="$BASE/envs"
LOG_ROOT="$BASE/logs/env_install"
PIP_CACHE_DIR="$PROJECT/.pip_cache"
export PIP_CACHE_DIR PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1

cd "$PROJECT"
mkdir -p "$ENV_ROOT" "$LOG_ROOT" "$BASE/logs"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to build envs outside a100-01" >&2
  exit 2
fi

pick_python() {
  local requested="$1"
  if [[ -x "$requested" ]]; then
    echo "$requested requested"
  elif [[ -x /usr/bin/python3.11 ]]; then
    echo "/usr/bin/python3.11 fallback_from_$requested"
  elif [[ -x /usr/bin/python3.12 ]]; then
    echo "/usr/bin/python3.12 fallback_from_$requested"
  else
    echo "python3 fallback_from_$requested"
  fi
}

build_env() {
  local repo="$1"
  local envname="$2"
  local note="$3"
  shift 3
  local requested=/usr/bin/python3.10
  local picked
  picked="$(pick_python "$requested")"
  local py="${picked%% *}"
  local py_status="${picked#* }"
  local envdir="$ENV_ROOT/$envname"
  local log="$LOG_ROOT/$envname.log"
  local envpy="$envdir/bin/python"

  echo "==== build $repo :: $envname"
  rm -f "$log"
  if [[ ! -x "$envpy" ]]; then
    "$py" -m venv "$envdir" >> "$log" 2>&1
    create_status=created
  else
    create_status=skipped_existing
  fi

  "$envpy" -m pip install -U "pip" "setuptools<81" "wheel" >> "$log" 2>&1
  "$envpy" -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 >> "$log" 2>&1
  "$envpy" -m pip install "$@" >> "$log" 2>&1

  "$envpy" - <<'PY'
import importlib.metadata as m
import torch
for p in ["torch", "torchvision", "diffusers", "transformers", "accelerate", "huggingface-hub", "tokenizers"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "MISSING")
print("torch_cuda", torch.version.cuda, torch.cuda.is_available())
PY

  python3 - "$BASE/e2_baseline_env_registry.csv" "$repo" "$envname" "$envdir" "$py" "$requested" "$py_status" "$envpy" "$create_status" "$log" "$note" <<'PY'
import csv
import sys
from pathlib import Path

registry = Path(sys.argv[1])
repo, envname, envdir, py, requested, py_status, envpy, create_status, log, note = sys.argv[2:]
fields = [
    "repo_name", "env_name", "env_path", "python", "requested_python",
    "python_status", "env_python", "create_status", "install_status",
    "log_path", "note",
]
rows = []
if registry.exists() and registry.stat().st_size:
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
rows = [row for row in rows if row.get("env_name") != envname and row.get("repo_name") != repo]
rows.append({
    "repo_name": repo,
    "env_name": envname,
    "env_path": envdir,
    "python": py,
    "requested_python": requested,
    "python_status": py_status,
    "env_python": envpy,
    "create_status": create_status,
    "install_status": "ok",
    "log_path": log,
    "note": note,
})
with registry.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY
}

build_env \
  "OT-RF" \
  "ot-rf-otip-py310" \
  "OT-RF/OTIP at abca084; FLUX/SD3 FlowEdit-style entrypoint." \
  diffusers==0.30.1 \
  transformers==4.46.3 \
  accelerate==1.0.1 \
  tokenizers==0.20.3 \
  huggingface-hub==0.29.1 \
  safetensors==0.5.3 \
  sentencepiece==0.2.0 \
  protobuf==5.29.3 \
  pyyaml==6.0.2 \
  pillow==9.5.0 \
  numpy==1.24.4 \
  scipy==1.10.1 \
  scikit-learn==1.3.2 \
  matplotlib==3.7.5 \
  tqdm==4.67.1

build_env \
  "DeltaRectifiedFlowSampling" \
  "dvrf-py310" \
  "Delta Rectified Flow Sampling at 567b28b; pip subset from drfs_environment.yml." \
  diffusers==0.30.1 \
  transformers==4.46.3 \
  accelerate==1.0.1 \
  tokenizers==0.20.3 \
  huggingface-hub==0.29.1 \
  safetensors==0.5.3 \
  sentencepiece==0.2.0 \
  protobuf==5.29.3 \
  pyyaml==6.0.2 \
  pillow==9.5.0 \
  numpy==1.24.4 \
  scipy==1.10.1 \
  scikit-learn==1.3.2 \
  matplotlib==3.7.5 \
  tqdm==4.67.1 \
  gpustat==1.1.1 \
  torchmetrics==1.5.2

echo build_new_e2_baseline_envs_complete
