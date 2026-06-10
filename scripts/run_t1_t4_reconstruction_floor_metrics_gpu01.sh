#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J t1t4-floor
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/t1_t4_floor_metrics_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/t1_t4_floor_metrics_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
PY="$PROJECT/_baselines/envs/fireflow-py310/bin/python"
TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run T1-T4 floor metrics outside a100-01" >&2
  exit 2
fi

"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/pretty_matrix" \
  --csv-output "$EXP/e4_t1_t4_reconstruction_floor_metrics.csv" \
  --json-output "$EXP/e4_t1_t4_reconstruction_floor_metrics.json" \
  --task-names "$TASKS" \
  --method-names "base_only" \
  --seeds "10 11 12" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks"
