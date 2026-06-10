#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=00:45:00
#SBATCH -J repair-rem
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/repair_remaining_envs_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/repair_remaining_envs_%j.err

set -u

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
LOG_DIR="$BASE/logs/repair_remaining_envs_${SLURM_JOB_ID:-manual}"
SUMMARY="$LOG_DIR/summary.txt"

mkdir -p "$LOG_DIR"
cd "$PROJECT" || exit 1

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to repair baseline envs outside a100-01" >&2
  exit 2
fi

touch "$SUMMARY"

run_step() {
  local name="$1"
  local log="$LOG_DIR/$name.log"
  shift
  echo "==== $name" > "$log"
  echo "cmd=$*" >> "$log"
  if "$@" >> "$log" 2>&1; then
    echo "$name,ok,$log" >> "$SUMMARY"
    echo "[$name] ok"
  else
    local status=$?
    echo "$name,failed:$status,$log" >> "$SUMMARY"
    echo "[$name] failed:$status"
  fi
}

run_step instruct_pix2pix_pin \
  "$BASE/envs/instruct-pix2pix-py39/bin/python" -m pip install \
  "diffusers==0.21.4" "transformers==4.33.3" \
  "tokenizers==0.13.3" "huggingface-hub==0.17.3" "accelerate==0.23.0"

run_step ledits_pp_editable \
  "$BASE/envs/ledits-pp-py310/bin/python" -m pip install \
  --no-build-isolation -e "$SRC/ledits_pp"

run_step instruct_pix2pix_import \
  "$BASE/envs/instruct-pix2pix-py39/bin/python" - <<'PY'
import diffusers
from diffusers import StableDiffusionInstructPix2PixPipeline
print("instruct-pix2pix import ok", diffusers.__version__)
PY

run_step ledits_pp_import \
  "$BASE/envs/ledits-pp-py310/bin/python" - <<'PY'
import diffusers
import leditspp
print("ledits_pp import ok", diffusers.__version__)
PY

echo "summary=$SUMMARY"
cat "$SUMMARY"
