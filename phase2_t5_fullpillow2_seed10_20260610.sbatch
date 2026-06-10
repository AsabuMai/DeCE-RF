#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:30:00
#SBATCH -J t5full2
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_fullpillow2_seed10_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_fullpillow2_seed10_%j.err

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

export T5_DECAL_BOX="0.195,0.300,0.558,0.830"
export T5_DECAL_SLANT_X="-0.02"
export T5_DECAL_PERSPECTIVE_Y="0.020"

export TARGET_PROMPT_OVERRIDE="A photo of the same white pillow on the same brown sofa, with the entire pillow surface changed into same-color white quilted honeycomb fabric with softly raised padded cells and visible seam grooves, while preserving the pillow shape, outline, lighting, sofa, background, and the rest of the scene."

run_variant() {
  local suffix="$1"; shift
  echo "[t5full2] ==== variant ${suffix}: $* ===="
  (
    export METHOD_NAME_SUFFIX="${suffix}"
    for kv in "$@"; do
      export "${kv?}"
    done
    bash scripts/run_pretty_matrix.sh
  )
}

# vR: vO recipe + strong final source-chroma projection.
run_variant "_t5vR" \
  DECAL_MATERIAL_CONTRAST=2.6 DECAL_MATERIAL_SCALE=1.5 \
  T5_FINAL_REF_COMPOSITE_SCALE=0.24 T5_FINAL_REF_COMPOSITE_BLUR=2.0 \
  T5_OPFIELD_HEDIT=0.50 T5_OPFIELD_TEXT=0.10 T5_OPFIELD_REC=0.46 T5_OPFIELD_TRAJ=0.28 \
  T5_REF_GUIDANCE_SCALE=0.40 T5_REF_LUMA_PRESERVE=0.30 T5_REF_GRADIENT_PRESERVE=0.10 \
  T5_FINAL_CHROMA_SOURCE_SCALE=0.85 T5_FINAL_CHROMA_SOURCE_BLUR=2.0

# vS: larger cells (vQ recipe) + chroma projection.
run_variant "_t5vS" \
  DECAL_MATERIAL_CONTRAST=2.6 DECAL_MATERIAL_SCALE=1.9 \
  T5_FINAL_REF_COMPOSITE_SCALE=0.24 T5_FINAL_REF_COMPOSITE_BLUR=2.0 \
  T5_OPFIELD_HEDIT=0.50 T5_OPFIELD_TEXT=0.10 T5_OPFIELD_REC=0.46 T5_OPFIELD_TRAJ=0.28 \
  T5_REF_GUIDANCE_SCALE=0.40 T5_REF_LUMA_PRESERVE=0.30 T5_REF_GRADIENT_PRESERVE=0.10 \
  T5_FINAL_CHROMA_SOURCE_SCALE=0.85 T5_FINAL_CHROMA_SOURCE_BLUR=2.0

# vT: vO recipe + full source-chroma replacement inside the panel.
run_variant "_t5vT" \
  DECAL_MATERIAL_CONTRAST=2.6 DECAL_MATERIAL_SCALE=1.5 \
  T5_FINAL_REF_COMPOSITE_SCALE=0.24 T5_FINAL_REF_COMPOSITE_BLUR=2.0 \
  T5_OPFIELD_HEDIT=0.50 T5_OPFIELD_TEXT=0.10 T5_OPFIELD_REC=0.46 T5_OPFIELD_TRAJ=0.28 \
  T5_REF_GUIDANCE_SCALE=0.40 T5_REF_LUMA_PRESERVE=0.30 T5_REF_GRADIENT_PRESERVE=0.10 \
  T5_FINAL_CHROMA_SOURCE_SCALE=1.0 T5_FINAL_CHROMA_SOURCE_BLUR=2.0

date
echo "[t5full2] sweep done"
