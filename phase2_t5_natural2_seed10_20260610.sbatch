#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:30:00
#SBATCH -J t5nat2
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_natural2_seed10_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_natural2_seed10_%j.err

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

run_variant() {
  local suffix="$1"; shift
  echo "[t5nat2] ==== variant ${suffix}: $* ===="
  (
    export METHOD_NAME_SUFFIX="${suffix}"
    for kv in "$@"; do
      export "${kv?}"
    done
    bash scripts/run_pretty_matrix.sh
  )
}

# vL: vI recipe with highlight-capped reference (shadow-led texture).
run_variant "_t5vL" \
  DECAL_MATERIAL_CONTRAST=1.6 DECAL_MATERIAL_SCALE=1.3 \
  T5_FINAL_REF_COMPOSITE_SCALE=0.18 T5_FINAL_REF_COMPOSITE_BLUR=2.5 \
  T5_OPFIELD_HEDIT=0.45 T5_OPFIELD_TEXT=0.08 T5_OPFIELD_REC=0.50 T5_OPFIELD_TRAJ=0.30 \
  T5_REF_GUIDANCE_SCALE=0.32 T5_REF_LUMA_PRESERVE=0.40 T5_REF_GRADIENT_PRESERVE=0.12

# vM: deeper shadow grooves, slightly stronger guidance.
run_variant "_t5vM" \
  DECAL_MATERIAL_CONTRAST=2.2 DECAL_MATERIAL_SCALE=1.4 \
  T5_FINAL_REF_COMPOSITE_SCALE=0.20 T5_FINAL_REF_COMPOSITE_BLUR=2.5 \
  T5_OPFIELD_HEDIT=0.50 T5_OPFIELD_TEXT=0.10 T5_OPFIELD_REC=0.46 T5_OPFIELD_TRAJ=0.28 \
  T5_REF_GUIDANCE_SCALE=0.38 T5_REF_LUMA_PRESERVE=0.32 T5_REF_GRADIENT_PRESERVE=0.10

# vN: no composite, shadow-led reference with stronger ODE pull.
run_variant "_t5vN" \
  DECAL_MATERIAL_CONTRAST=2.2 DECAL_MATERIAL_SCALE=1.4 \
  T5_FINAL_REF_COMPOSITE_SCALE=0.0 \
  T5_OPFIELD_HEDIT=0.55 T5_OPFIELD_TEXT=0.10 T5_OPFIELD_REC=0.42 T5_OPFIELD_TRAJ=0.25 \
  T5_REF_GUIDANCE_SCALE=0.45 T5_REF_LUMA_PRESERVE=0.30 T5_REF_GRADIENT_PRESERVE=0.08

date
echo "[t5nat2] sweep done"
