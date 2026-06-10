#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:30:00
#SBATCH -J t6fill
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t6_fill_probes_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/phase2_t6_fill_probes_%j.err

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
export TASKS="backpack_remove_toy_charm"
export METHODS="support_v3_controller_rmsgap"
export SEEDS="10"
export MAX_IMAGE_SIZE=512
export NUM_INFERENCE_STEPS=28
export REGENERATE_MASKS=1
export SKIP_EXISTING=0

TEXTURE_PROMPT="A close-up photo of the same grey backpack with the yellow dangling toy charm and its cast shadow completely removed, leaving only the plain pebbled grey backpack fabric with its natural grain and seams, with the pink strap, zipper, and all other details preserved."

run_variant() {
  local suffix="$1"; shift
  echo "[t6fill] ==== variant ${suffix}: $* ===="
  (
    export METHOD_NAME_SUFFIX="${suffix}"
    for kv in "$@"; do
      export "${kv?}"
    done
    bash scripts/run_pretty_matrix.sh
  )
}

# v6A: enable the clean_fill removal controller, conservative scales.
run_variant "_t6vA" \
  SUPPORT_V3_REMOVAL_CONTROLLER_MODE=clean_fill \
  SUPPORT_V3_REMOVAL_FILL_SCALE=0.50 SUPPORT_V3_REMOVAL_SUPPRESSION_SCALE=0.25 SUPPORT_V3_REMOVAL_RING_REC_SCALE=0.40

# v6B: clean_fill defaults + dilated mask to cover the cast shadow.
run_variant "_t6vB" \
  SUPPORT_V3_REMOVAL_CONTROLLER_MODE=clean_fill \
  SUPPORT_V3_REMOVAL_FILL_SCALE=0.70 SUPPORT_V3_REMOVAL_SUPPRESSION_SCALE=0.35 SUPPORT_V3_REMOVAL_RING_REC_SCALE=0.40 \
  T6_SEMANTIC_DILATE=16

# v6C: v6B + texture/shadow-aware prompt.
run_variant "_t6vC" \
  SUPPORT_V3_REMOVAL_CONTROLLER_MODE=clean_fill \
  SUPPORT_V3_REMOVAL_FILL_SCALE=0.70 SUPPORT_V3_REMOVAL_SUPPRESSION_SCALE=0.35 SUPPORT_V3_REMOVAL_RING_REC_SCALE=0.40 \
  T6_SEMANTIC_DILATE=16 \
  "TARGET_PROMPT_OVERRIDE=${TEXTURE_PROMPT}"

date
echo "[t6fill] probes done"
