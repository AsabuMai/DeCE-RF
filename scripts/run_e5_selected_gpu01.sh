#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=04:00:00
#SBATCH -J e5-select
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e5_selected_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e5_selected_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
PY="$PROJECT/.venv/bin/python"
cd "$PROJECT"

hostname
date
nvidia-smi || true

BASE_TASKS="laptop_remove_sticker fridge_remove_yellow_magnet fridge_remove_peach_magnet whiteboard_remove_yellow_letter dog_remove_tennis_ball dog_replace_tennis_ball_star"
REMOVAL_TASKS="laptop_remove_sticker fridge_remove_yellow_magnet fridge_remove_peach_magnet whiteboard_remove_yellow_letter dog_remove_tennis_ball"
SEEDS="10 11 12"

export TASKS="$BASE_TASKS"
export METHODS="support_v3_controller_rmsgap"
export SEEDS="$SEEDS"
export SKIP_EXISTING=1
export REUSE_SEMANTIC_MASKS=1
export REGENERATE_MASKS=0
bash scripts/run_pretty_matrix.sh

"$PY" scripts/run_gated_clean_delta_experiment.py \
  --root "$PROJECT" \
  --tasks "$REMOVAL_TASKS" \
  --seeds "$SEEDS" \
  --source-method support_v3_controller_rmsgap \
  --output-method support_v3_controller_rmsgap_completion_clean_delta_gated_highconf \
  --reliability-csv experiments/support_v3_2026-05-11/prior_reliability/completion_prior_reliability_seed10.csv \
  --protocol-output experiments/support_v3_2026-06-02/e5_boundary_extension/e5_gated_completion_protocol.json \
  --skip-existing

"$PY" scripts/run_whiteboard_replacement_probe.py \
  --root "$PROJECT" \
  --seeds "$SEEDS" \
  --variants whiteboard_probe_red_star_sticker \
  --methods support_v3_controller_rmsgap_replace_editor_v1 \
  --protocol-output experiments/support_v3_2026-06-02/e5_boundary_extension/e5_whiteboard_red_star_protocol.json \
  --skip-existing

date
