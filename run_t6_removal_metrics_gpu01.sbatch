#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J t6metrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/t6_removal_metrics_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/t6_removal_metrics_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
PY="$PROJECT/_baselines/envs/fireflow-py310/bin/python"

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
  --csv-output "$EXP/t6_removal_aware_metrics.csv" \
  --json-output "$EXP/t6_removal_aware_metrics.json" \
  --task-names "backpack_remove_toy_charm" \
  --method-names "base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap" \
  --seeds "10 11 12" \
  --eval-mask-dir "$EXP/eval_masks" \
  --clip-model openai/clip-vit-large-patch14 \
  --removal-phrase "backpack_remove_toy_charm=a yellow cartoon toy charm figurine|plain grey backpack fabric"

date
echo "[t6metrics] removal-aware metrics done"
