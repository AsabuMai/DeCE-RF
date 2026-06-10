#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J t5or
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_outside_restore_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_outside_restore_%j.err

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
export REGENERATE_MASKS=1
export SKIP_EXISTING=0

# Uniform recipe re-run with pixel-space outside restore on all cable-knit rows.
export TASKS="pillow_same_color_cable_knit pillow_same_color_cable_knit_grey"
export SEEDS="10 11 12"
bash scripts/run_pretty_matrix.sh

export TASKS="pillow_same_color_cable_knit_armchair"
export SEEDS="10"
bash scripts/run_pretty_matrix.sh

date
echo "[t5or] outside-restore rerun done"
