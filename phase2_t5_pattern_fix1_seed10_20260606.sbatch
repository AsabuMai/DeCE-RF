#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J t5_fix1
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_pattern_fix1_seed10_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/phase2_t5_pattern_fix1_seed10_%j.er

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

export TASKS="white_pillow_blue_dots_phase2 white_pillow_blue_cross_phase2"
export METHODS="support_v3_controller_rmsgap"
export SEEDS="10"
export METHOD_NAME_SUFFIX="_t5fix1"
export SKIP_EXISTING=0
export REGENERATE_MASKS=1
export REUSE_SEMANTIC_MASKS=1
export SEMANTIC_MASK_CACHE_METHOD="support_v3_controller_rmsgap"

bash scripts/run_pretty_matrix.sh

date
