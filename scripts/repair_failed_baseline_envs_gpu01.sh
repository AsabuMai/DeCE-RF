#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J repair-baselines
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/repair_failed_envs_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/repair_failed_envs_%j.err

set -u

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
LOG_DIR="$BASE/logs/repair_failed_envs_${SLURM_JOB_ID:-manual}"
SUMMARY="$LOG_DIR/summary.txt"
HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export HF_HUB_CACHE

mkdir -p "$LOG_DIR" "$HF_HUB_CACHE"
cd "$PROJECT" || exit 1

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to repair GPU baseline envs outside a100-01" >&2
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

download_zone_sam() {
  local target="$SRC/ZONE/ckpts/sam_vit_h_4b8939.pth"
  mkdir -p "$(dirname "$target")"
  if [[ -s "$target" ]]; then
    echo "SAM checkpoint already exists: $target"
    return 0
  fi
  curl -L --fail --retry 3 \
    -o "$target.tmp" \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  mv "$target.tmp" "$target"
  test -s "$target"
}

run_step zone_sam download_zone_sam

run_step instruct_pix2pix_versions \
  "$BASE/envs/instruct-pix2pix-py39/bin/python" -m pip install \
  "transformers==4.46.3" "tokenizers==0.20.3"

run_step prompt_to_prompt_versions \
  "$BASE/envs/prompt-to-prompt-py310/bin/python" -m pip install \
  "diffusers==0.3.0" "transformers==4.26.1" \
  "tokenizers==0.13.3" "huggingface-hub==0.13.4"

run_step h_edit_versions \
  "$BASE/envs/h-edit-py310/bin/python" -m pip install \
  "diffusers==0.21.4" "transformers==4.33.3" \
  "huggingface-hub==0.17.3" "accelerate==0.23.0"

run_step ledits_pp_versions \
  "$BASE/envs/ledits-pp-py310/bin/python" -m pip install -e "$SRC/ledits_pp"

run_step pix2pix_zero_versions \
  "$BASE/envs/pix2pix-zero-py310/bin/python" -m pip install \
  "diffusers==0.12.0" "transformers==4.26.1" \
  "tokenizers==0.13.3" "huggingface-hub==0.13.4" "accelerate==0.18.0"

run_step masactrl_versions \
  "$BASE/envs/masactrl-py310/bin/python" -m pip install \
  "diffusers==0.21.4" "transformers==4.33.3" \
  "huggingface-hub==0.17.3" "accelerate==0.23.0"

run_step instruct_pix2pix_import \
  "$BASE/envs/instruct-pix2pix-py39/bin/python" - <<'PY'
from diffusers import StableDiffusionInstructPix2PixPipeline
from transformers import SiglipImageProcessor
print("instruct-pix2pix import ok")
PY

run_step prompt_to_prompt_import \
  "$BASE/envs/prompt-to-prompt-py310/bin/python" - <<'PY'
from diffusers import StableDiffusionPipeline
import transformers
print("prompt-to-prompt import ok", transformers.__version__)
PY

run_step h_edit_import \
  "$BASE/envs/h-edit-py310/bin/python" - <<'PY'
import diffusers
from diffusers import StableDiffusionPipeline
print("h-edit import ok", diffusers.__version__)
PY

run_step ledits_pp_import \
  "$BASE/envs/ledits-pp-py310/bin/python" - <<'PY'
import diffusers
print("ledits_pp import ok", diffusers.__version__)
PY

run_step pix2pix_zero_import \
  "$BASE/envs/pix2pix-zero-py310/bin/python" - <<'PY'
import diffusers
from diffusers import StableDiffusionPipeline
print("pix2pix-zero import ok", diffusers.__version__)
PY

run_step masactrl_import \
  "$BASE/envs/masactrl-py310/bin/python" - <<'PY'
from diffusers import StableDiffusionXLPipeline
print("MasaCtrl SDXL import ok")
PY

echo "summary=$SUMMARY"
cat "$SUMMARY"
