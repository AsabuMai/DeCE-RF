#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:30:00
#SBATCH -J t5bn
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_boundary_naturalness_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_boundary_naturalness_%j.err

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
export METHODS="support_v3_controller_rmsgap"
export MAX_IMAGE_SIZE=512
export NUM_INFERENCE_STEPS=28
export SKIP_EXISTING=0

# Armchair: rerun seed 10 with tightened bright-host mask (erode + intersect).
(
  export TASKS="pillow_same_color_cable_knit_armchair"
  export SEEDS="10"
  export REGENERATE_MASKS=1
  bash scripts/run_pretty_matrix.sh
)

# Grey: try seeds 11/12 for a better shadow-side knit rendering.
(
  export TASKS="pillow_same_color_cable_knit_grey"
  export SEEDS="11 12"
  export REGENERATE_MASKS=1
  bash scripts/run_pretty_matrix.sh
)

date
echo "[t5bn] boundary/naturalness pass done"
