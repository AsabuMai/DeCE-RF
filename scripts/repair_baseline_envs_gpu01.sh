#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J repair-e2-baselines
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/env_repair_20260603/repair_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/env_repair_20260603/repair_%j.err

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

torch_envs=(
  fireflow-py310
  flowalign-py310
  flowedit-py310
  splitflow-py310
  rf-solver-edit-py310
  stable-flow-py311
  zone-py310
  instruct-pix2pix-py39
  pix2pix-zero-py310
  masactrl-py310
  prompt-to-prompt-py310
  h-edit-py310
  ledits-pp-py310
)

torch_state() {
  local envname="$1"
  "_baselines/envs/$envname/bin/python" - <<'PY'
import importlib.metadata as m
try:
    import torch
    print(m.version("torch"), torch.version.cuda, torch.cuda.is_available())
except Exception as exc:
    print("ERR", type(exc).__name__, exc)
PY
}

repair_torch() {
  local envname="$1"
  local py="_baselines/envs/$envname/bin/python"
  local before
  before="$(torch_state "$envname" | tail -n 1)"
  echo "==== torch check: $envname :: $before"
  if [[ "$before" == 2.4.1*' 12.1 True' ]]; then
    echo "torch already usable for $envname"
    return
  fi
  "$py" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    > "$LOGDIR/${envname}_torch_cu121.log" 2>&1
  echo "after: $(torch_state "$envname" | tail -n 1)"
}

pin_install() {
  local envname="$1"
  shift
  local py="_baselines/envs/$envname/bin/python"
  echo "==== HF/version repair: $envname :: $*"
  "$py" -m pip install --upgrade "$@" > "$LOGDIR/${envname}_hf_pins.log" 2>&1
  "$py" - <<'PY'
import importlib.metadata as m
for p in ["diffusers", "transformers", "accelerate", "huggingface-hub", "tokenizers"]:
    try:
        print(p, m.version(p))
    except Exception:
        print(p, "MISSING")
PY
}

for envname in "${torch_envs[@]}"; do
  repair_torch "$envname"
done

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

echo "repair_complete"
