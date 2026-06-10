#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J p0e2fin
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/p0_e2_finish_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/p0_e2_finish_%j.err

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

# Stage 1: re-run the E2 T1-T4 external baseline metrics with CLIP+DINO
# (fills the empty edit-score column; CLIP cache fixed 2026-06-10).
bash scripts/run_e2_t1_t4_baseline_metrics_gpu01.sh

date
echo "[p0e2] baseline metrics with CLIP done"

# Stage 2: missing Phase 2 internal-method metric rows (consolidated-report
# gap): direct_target and generic support on the 12 Phase 2 T1-T4 tasks,
# same protocol as the e4 T1-T4 metrics.
TASKS_P2="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"

"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/pretty_matrix" \
  --csv-output "$EXP/e1_t1_t4_directgeneric_metrics.csv" \
  --json-output "$EXP/e1_t1_t4_directgeneric_metrics.json" \
  --task-names "$TASKS_P2" \
  --method-names "direct_target adaptive_full_generic_support" \
  --seeds "10 11 12" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --preserve-floor-csv "$EXP/e4_t1_t4_reconstruction_floor_metrics.csv" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --allow-download

date
echo "[p0e2] phase2 direct/generic gap-fill done"
