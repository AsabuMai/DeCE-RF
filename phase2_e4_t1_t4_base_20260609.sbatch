#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=12:00:00
#SBATCH -J e4-t1t4-base
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/e4_t1_t4_base_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/e4_t1_t4_base_%j.err

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

export TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"
export METHODS="support_v3_fixed support_v3_controller_rmsgap"
export SEEDS="10 11 12"
export SKIP_EXISTING=1
export REUSE_SEMANTIC_MASKS=1
export SEMANTIC_MASK_CACHE_METHOD="support_v3_controller_rmsgap"

bash scripts/run_pretty_matrix.sh

date
