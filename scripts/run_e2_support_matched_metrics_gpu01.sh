#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J e2-4-metrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_support_matched_metrics_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_support_matched_metrics_%j.err

set -u

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
OUT="$PROJECT/outputs/e2_support_matched_diagnostic"
PY="$PROJECT/_baselines/envs/fireflow-py310/bin/python"

cd "$PROJECT" || exit 1

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run E2.4 metrics outside a100-01" >&2
  exit 2
fi

"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$OUT" \
  --csv-output "$EXP/e2_support_matched_fixed_mask_metrics.csv" \
  --json-output "$EXP/e2_support_matched_fixed_mask_metrics.json" \
  --task-names "cat_crown tshirt_star backpack_remove_toy_charm" \
  --method-names "direct_target_raw direct_target_mask_blend flowedit_mask_blend support_v3_controller_rmsgap" \
  --seeds "10 11" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --preserve-floor-csv "$EXP/strict_fixed_mask_metrics.csv"

"$PY" scripts/summarize_e2_support_matched_diagnostic.py
