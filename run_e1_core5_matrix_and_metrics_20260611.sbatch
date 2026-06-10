#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=03:00:00
#SBATCH -J e1core5
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/e1_core5_matrix_metrics_%j.log
#SBATCH -e /cluster/users/grad/2025/25t8103/project/logs/e1_core5_matrix_metrics_%j.err

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

# Stage 1: generate the 9 missing cable_knit baseline-method runs.
export PYTHON="$PY"
export TASKS="pillow_same_color_cable_knit"
export METHODS="base_only direct_target adaptive_full_generic_support"
export SEEDS="10 11 12"
export MAX_IMAGE_SIZE=512
export NUM_INFERENCE_STEPS=28
export SKIP_EXISTING=1
bash scripts/run_pretty_matrix.sh

date
echo "[e1core5] generation stage done"

# Stage 2: full Core-5 metric pass for Table 1.
"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/pretty_matrix" \
  --csv-output "$EXP/e1_core5_strict_metrics.csv" \
  --json-output "$EXP/e1_core5_strict_metrics.json" \
  --task-names "cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_same_color_cable_knit" \
  --method-names "base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap" \
  --seeds "10 11 12" \
  --eval-mask-dir "$EXP/eval_masks" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --allow-download \
  --success-spec "cat_crown=a small golden crown|cat|above_host" \
  --success-spec "bowl_apple_inside=a red apple|blue ceramic bowl|inside_host" \
  --success-spec "tshirt_star=a red star shape|white t-shirt|on_host"

date
echo "[e1core5] metric pass done"
