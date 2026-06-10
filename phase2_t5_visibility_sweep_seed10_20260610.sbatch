#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J t5vis
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_visibility_sweep_seed10_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_visibility_sweep_seed10_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

export PYTHON=.venv/bin/python
export TASKS="pillow_same_color_corduroy_panel"
export METHODS="support_v3_controller_rmsgap"
export SEEDS="10"
export MAX_IMAGE_SIZE=512
export NUM_INFERENCE_STEPS=28
export REGENERATE_MASKS=1
export SKIP_EXISTING=0

STRONG_PROMPT="A photo of the same plain white pillow on the same brown sofa, with only the center vertical surface panel changed into same-color white corduroy fabric with wide raised vertical ribs and clearly visible shadowed grooves between the ribs, while preserving the pillow shape, folds, lighting, contour, sofa, background, and all surrounding pillow fabric."

run_variant() {
  local suffix="$1"; shift
  echo "[t5vis] ==== variant ${suffix}: $* ===="
  (
    export METHOD_NAME_SUFFIX="${suffix}"
    for kv in "$@"; do
      export "${kv?}"
    done
    bash scripts/run_pretty_matrix.sh
  )
}

run_variant "_t5vA" \
  DECAL_MATERIAL_CONTRAST=1.8 DECAL_MATERIAL_SCALE=1.8 \
  T5_REF_GUIDANCE_SCALE=0.50 T5_REF_LUMA_PRESERVE=0.45 T5_REF_GRADIENT_PRESERVE=0.18

run_variant "_t5vB" \
  DECAL_MATERIAL_CONTRAST=2.4 DECAL_MATERIAL_SCALE=2.2 \
  T5_REF_GUIDANCE_SCALE=0.60 T5_REF_LUMA_PRESERVE=0.32 T5_REF_GRADIENT_PRESERVE=0.12 \
  T5_OPFIELD_REC=0.38

run_variant "_t5vC" \
  DECAL_MATERIAL_CONTRAST=3.0 DECAL_MATERIAL_SCALE=2.6 \
  T5_REF_GUIDANCE_SCALE=0.72 T5_REF_LUMA_PRESERVE=0.20 T5_REF_GRADIENT_PRESERVE=0.08 \
  T5_OPFIELD_REC=0.30 T5_OPFIELD_HEDIT=0.66

run_variant "_t5vD" \
  DECAL_MATERIAL_CONTRAST=2.4 DECAL_MATERIAL_SCALE=2.2 \
  T5_REF_GUIDANCE_SCALE=0.60 T5_REF_LUMA_PRESERVE=0.32 T5_REF_GRADIENT_PRESERVE=0.12 \
  T5_OPFIELD_REC=0.38 "T5_TARGET_PROMPT_OVERRIDE=${STRONG_PROMPT}"

date
echo "[t5vis] sweep done"
