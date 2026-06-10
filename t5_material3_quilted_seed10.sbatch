#!/bin/bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -J t5quilt
#SBATCH -o logs/t5quilt_%j.out
#SBATCH -e logs/t5quilt_%j.err

set -euo pipefail
hostname
cd /cluster/users/grad/2025/25t8103/project
export PYTHON=.venv/bin/python
export TASKS="pillow_same_color_corduroy_panel pillow_same_color_linen_panel pillow_same_color_terry_panel"
export METHODS=support_v3_controller_rmsgap
export SEEDS=10
export MAX_IMAGE_SIZE=512
export NUM_INFERENCE_STEPS=28
export REGENERATE_MASKS=1
bash scripts/run_pretty_matrix.sh
