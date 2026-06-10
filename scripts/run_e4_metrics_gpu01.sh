#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J e4-metrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e4_metrics_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e4_metrics_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
PY="$PROJECT/_baselines/envs/fireflow-py310/bin/python"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run outside a100-01" >&2
  exit 2
fi

"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/pretty_matrix" \
  --csv-output "$EXP/e4_controller_base_metrics.csv" \
  --json-output "$EXP/e4_controller_base_metrics.json" \
  --task-names "cat_crown tshirt_star pillow_vertical_fabric_strip" \
  --method-names "support_v3_fixed support_v3_controller_rmsgap" \
  --seeds "10 11 12" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --preserve-floor-csv "$EXP/strict_fixed_mask_metrics.csv"

"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/pretty_matrix" \
  --csv-output "$EXP/e4_edit_strength_metrics.csv" \
  --json-output "$EXP/e4_edit_strength_metrics.json" \
  --task-names "cat_crown tshirt_star pillow_vertical_fabric_strip" \
  --method-names "support_v3_fixed support_v3_controller_rmsgap support_v3_fixed_e4x050 support_v3_controller_rmsgap_e4x050 support_v3_fixed_e4x075 support_v3_controller_rmsgap_e4x075 support_v3_fixed_e4x125 support_v3_controller_rmsgap_e4x125 support_v3_fixed_e4x150 support_v3_controller_rmsgap_e4x150 support_v3_fixed_e4x200 support_v3_controller_rmsgap_e4x200" \
  --seeds "10" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --preserve-floor-csv "$EXP/strict_fixed_mask_metrics.csv"
