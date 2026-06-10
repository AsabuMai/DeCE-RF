#!/bin/bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -J t4vase10
#SBATCH -o logs/t4vase10_%j.out
#SBATCH -e logs/t4vase10_%j.err

set -euo pipefail
hostname
cd /cluster/users/grad/2025/25t8103/project
export PYTHON=.venv/bin/python
export TASKS=yellow_vase_blue_phase2
export METHODS=support_v3_controller_rmsgap
export METHOD_NAME_SUFFIX=_vasecheck_seed10
export SEEDS=10
export MAX_IMAGE_SIZE=512
export NUM_INFERENCE_STEPS=28
export REGENERATE_MASKS=1
bash scripts/run_pretty_matrix.sh
