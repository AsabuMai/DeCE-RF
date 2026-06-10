#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:30:00
#SBATCH -J t5vis2
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_visibility_fix2_seed10_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_visibility_fix2_seed10_%j.err

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
  echo "[t5vis2] ==== variant ${suffix}: $* ===="
  (
    export METHOD_NAME_SUFFIX="${suffix}"
    for kv in "$@"; do
      export "${kv?}"
    done
    bash scripts/run_pretty_matrix.sh
  )
}

# vE: quilted luma bugfix only, embedded task defaults.
run_variant "_t5vE" \
  DECAL_MATERIAL_CONTRAST=1.0 DECAL_MATERIAL_SCALE=1.0

# vF: moderately deeper and larger cells.
run_variant "_t5vF" \
  DECAL_MATERIAL_CONTRAST=1.6 DECAL_MATERIAL_SCALE=1.3

# vG: strongest texture push plus slightly stronger ref guidance.
run_variant "_t5vG" \
  DECAL_MATERIAL_CONTRAST=2.2 DECAL_MATERIAL_SCALE=1.6 \
  T5_REF_GUIDANCE_SCALE=0.30 T5_REF_LUMA_PRESERVE=0.12

date
echo "[t5vis2] sweep done"
