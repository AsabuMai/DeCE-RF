#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J t6fill4
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t6_fill_probes4_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/phase2_t6_fill_probes4_%j.err

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

export T6_SEMANTIC_DILATE=16
export T6_FILL_PATCH_BOX="0.45,0.94,0.85,1.0"
export T6_FILL_GRAIN_SCALE=1.1

run_variant() {
  local suffix="$1"; shift
  echo "[t6fill4] ==== variant ${suffix}: $* ===="
  (
    export METHOD_NAME_SUFFIX="${suffix}"
    for kv in "$@"; do
      export "${kv?}"
    done
    bash scripts/run_pretty_matrix.sh
  )
}

# v6J: corrected structure image, moderate ref pull.
run_variant "_t6vJ" \
  T6_FILL_REF_SCALE=0.30

# v6K: stronger ref pull.
run_variant "_t6vK" \
  T6_FILL_REF_SCALE=0.50

date
echo "[t6fill4] probes done"
