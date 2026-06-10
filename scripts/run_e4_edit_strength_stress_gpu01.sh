#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=03:00:00
#SBATCH -J e4-edit-stress
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e4_edit_stress_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e4_edit_stress_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

export TASKS="cat_crown tshirt_star pillow_vertical_fabric_strip"
export METHODS="support_v3_fixed support_v3_controller_rmsgap"
export SEEDS="10"
export SKIP_EXISTING=1
export REUSE_SEMANTIC_MASKS=1
export SEMANTIC_MASK_CACHE_METHOD="support_v3_controller_rmsgap"

run_level() {
  local level="$1"
  local suffix="$2"
  echo "stress_level=$level suffix=$suffix"
  export EDIT_STRENGTH_MULTIPLIER="$level"
  export METHOD_NAME_SUFFIX="$suffix"
  bash scripts/run_pretty_matrix.sh
}

run_level "0.50" "_e4x050"
run_level "0.75" "_e4x075"
run_level "1.25" "_e4x125"
run_level "1.50" "_e4x150"
run_level "2.00" "_e4x200"
