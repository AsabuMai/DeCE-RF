#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:30:00
#SBATCH -J e4-fill-base
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e4_fill_base_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e4_fill_base_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
cd "$PROJECT"

hostname
date
nvidia-smi || true

export REUSE_SEMANTIC_MASKS=1
export SKIP_EXISTING=0

export TASKS="tshirt_star"
export METHODS="support_v3_fixed"
export SEEDS="12"
export SEMANTIC_MASK_CACHE_METHOD="support_v3_controller_rmsgap"
bash scripts/run_pretty_matrix.sh

export TASKS="pillow_vertical_fabric_strip"
export METHODS="support_v3_controller_rmsgap"
export SEEDS="10 11 12"
export SEMANTIC_MASK_CACHE_METHOD="support_v3_fixed"
bash scripts/run_pretty_matrix.sh

date
