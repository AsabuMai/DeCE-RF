#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J tsucc
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/task_success_validation_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/task_success_validation_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
PY="$PROJECT/.venv/bin/python"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
date
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/pretty_matrix" \
  --csv-output "$EXP/task_success_validation_metrics.csv" \
  --json-output "$EXP/task_success_validation_metrics.json" \
  --task-names "cat_crown bowl_apple_inside tshirt_star" \
  --method-names "base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap" \
  --seeds "10 11 12" \
  --eval-mask-dir "$EXP/eval_masks" \
  --success-spec "cat_crown=a small golden crown|cat|above_host" \
  --success-spec "bowl_apple_inside=a red apple|blue ceramic bowl|inside_host" \
  --success-spec "tshirt_star=a red star shape|white t-shirt|on_host"

date
echo "[tsucc] task-success validation done"
