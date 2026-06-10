#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:30:00
#SBATCH -J t5waff
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_waffle_seed10_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_waffle_seed10_%j.err

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

WAFFLE_PROMPT="A photo of the same white pillow on the same brown sofa, with the entire pillow surface changed into same-color white waffle-weave fabric with a crisp raised square grid texture, while preserving the pillow shape, outline, lighting, sofa, background, and the rest of the scene."
KNIT_PROMPT="A photo of the same white pillow on the same brown sofa, with the entire pillow surface changed into same-color white chunky cable-knit fabric with thick braided knitted columns, while preserving the pillow shape, outline, lighting, sofa, background, and the rest of the scene."

run_variant() {
  local suffix="$1"; shift
  echo "[t5waff] ==== variant ${suffix}: $* ===="
  (
    export METHOD_NAME_SUFFIX="${suffix}"
    for kv in "$@"; do
      export "${kv?}"
    done
    bash scripts/run_pretty_matrix.sh
  )
}

# vU: waffle weave, crisp seams, moderate composite, chroma projection.
run_variant "_t5vU" \
  T5_DECAL_SHAPE=waffle_panel \
  "TARGET_PROMPT_OVERRIDE=${WAFFLE_PROMPT}" \
  "ATTENTION_TARGET_WORDS_OVERRIDE=waffle,grid,weave,pillow" \
  "CHANGED_TARGET_WORDS_OVERRIDE=waffle,grid,weave" \
  "SUPPORT_NEW_TOKENS_OVERRIDE=waffle,grid,weave,material" \
  DECAL_MATERIAL_CONTRAST=1.0 DECAL_MATERIAL_SCALE=1.0 \
  T5_FINAL_REF_COMPOSITE_SCALE=0.22 T5_FINAL_REF_COMPOSITE_BLUR=1.2 \
  T5_OPFIELD_HEDIT=0.50 T5_OPFIELD_TEXT=0.10 T5_OPFIELD_REC=0.46 T5_OPFIELD_TRAJ=0.28 \
  T5_REF_GUIDANCE_SCALE=0.40 T5_REF_LUMA_PRESERVE=0.30 T5_REF_GRADIENT_PRESERVE=0.10 \
  T5_FINAL_CHROMA_SOURCE_SCALE=1.0 T5_FINAL_CHROMA_SOURCE_BLUR=2.0

# vV: waffle weave, larger cells and deeper seams.
run_variant "_t5vV" \
  T5_DECAL_SHAPE=waffle_panel \
  "TARGET_PROMPT_OVERRIDE=${WAFFLE_PROMPT}" \
  "ATTENTION_TARGET_WORDS_OVERRIDE=waffle,grid,weave,pillow" \
  "CHANGED_TARGET_WORDS_OVERRIDE=waffle,grid,weave" \
  "SUPPORT_NEW_TOKENS_OVERRIDE=waffle,grid,weave,material" \
  DECAL_MATERIAL_CONTRAST=1.4 DECAL_MATERIAL_SCALE=1.4 \
  T5_FINAL_REF_COMPOSITE_SCALE=0.26 T5_FINAL_REF_COMPOSITE_BLUR=1.2 \
  T5_OPFIELD_HEDIT=0.50 T5_OPFIELD_TEXT=0.10 T5_OPFIELD_REC=0.46 T5_OPFIELD_TRAJ=0.28 \
  T5_REF_GUIDANCE_SCALE=0.40 T5_REF_LUMA_PRESERVE=0.30 T5_REF_GRADIENT_PRESERVE=0.10 \
  T5_FINAL_CHROMA_SOURCE_SCALE=1.0 T5_FINAL_CHROMA_SOURCE_BLUR=2.0

# vW: cable-knit probe, model-driven (no composite, weak ref hint).
run_variant "_t5vW" \
  T5_DECAL_SHAPE=waffle_panel \
  "TARGET_PROMPT_OVERRIDE=${KNIT_PROMPT}" \
  "ATTENTION_TARGET_WORDS_OVERRIDE=knit,cable,braided,pillow" \
  "CHANGED_TARGET_WORDS_OVERRIDE=knit,cable,braided" \
  "SUPPORT_NEW_TOKENS_OVERRIDE=knit,cable,braided,material" \
  DECAL_MATERIAL_CONTRAST=1.0 DECAL_MATERIAL_SCALE=1.0 \
  T5_FINAL_REF_COMPOSITE_SCALE=0.0 \
  T5_OPFIELD_HEDIT=0.68 T5_OPFIELD_TEXT=0.14 T5_OPFIELD_REC=0.40 T5_OPFIELD_TRAJ=0.25 \
  T5_REF_GUIDANCE_SCALE=0.08 T5_REF_LUMA_PRESERVE=0.60 T5_REF_GRADIENT_PRESERVE=0.20 \
  T5_FINAL_CHROMA_SOURCE_SCALE=1.0 T5_FINAL_CHROMA_SOURCE_BLUR=2.0

date
echo "[t5waff] sweep done"
