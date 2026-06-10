#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J e2e5-t1t4-post
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/e2_e5_t1_t4_postprocess_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/e2_e5_t1_t4_postprocess_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
PY="$PROJECT/.venv/bin/python"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

"$PY" scripts/prepare_t1_t4_eval_assets.py
"$PY" scripts/prepare_e2_support_matched_diagnostic_t1_t4.py
"$PY" scripts/summarize_e3_support_geometry_t1_t4.py
"$PY" scripts/summarize_e5_t1_t4_boundary_extension.py

date
