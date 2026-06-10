#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=00:45:00
#SBATCH -J e2-4-tshirt-fixed
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_4_tshirt_fixed_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_4_tshirt_fixed_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

export TASKS="tshirt_star"
export METHODS="support_v3_fixed"
export SEEDS="10 11"
export SKIP_EXISTING=0
export REUSE_SEMANTIC_MASKS=1
export SEMANTIC_MASK_CACHE_METHOD="support_v3_controller_rmsgap"

bash scripts/run_pretty_matrix.sh
