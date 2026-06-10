#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=00:30:00
#SBATCH -J e4-post
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e4_postprocess_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e4_postprocess_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP=experiments/support_v3_2026-06-02
OUT="$EXP/e4_controller_ablation"
PY="$PROJECT/_baselines/envs/fireflow-py310/bin/python"

cd "$PROJECT"

hostname
date

python3 -m py_compile scripts/check_e4_base_files.py
python3 -m py_compile scripts/summarize_e4_controller_ablation.py

python3 scripts/check_e4_base_files.py
"$PY" scripts/summarize_e4_controller_ablation.py

mkdir -p "$OUT"

base_rows=$(($(wc -l < "$EXP/e4_controller_base_metrics.csv") - 1))
stress_rows=$(($(wc -l < "$EXP/e4_edit_strength_metrics.csv") - 1))
base_summary_rows=$(($(wc -l < "$OUT/e4_controller_base_summary.csv") - 1))
stress_summary_rows=$(($(wc -l < "$OUT/e4_edit_strength_summary.csv") - 1))
trajectory_rows=$(($(wc -l < "$OUT/e4_controller_trajectory_stats.csv") - 1))

cat > "$OUT/e4_controller_ablation_complete_2026-06-04.csv" <<EOF
item,value
base_metric_rows,$base_rows
stress_metric_rows,$stress_rows
base_summary_rows,$base_summary_rows
stress_summary_rows,$stress_summary_rows
trajectory_rows,$trajectory_rows
tasks,cat_crown;tshirt_star;pillow_vertical_fabric_strip
base_methods,support_v3_fixed;support_v3_controller_rmsgap
stress_seed,10
stress_multipliers,0.50;0.75;1.00;1.25;1.50;2.00
backbone,Stable Diffusion 3 Medium / project RF edit pipeline
normalization_policy,512 evaluation masks and native pretty_matrix outputs
EOF

cat > "$OUT/e4_controller_ablation_complete_2026-06-04.md" <<EOF
# E4 Controller Ablation Completion

Date: 2026-06-04

E4 compares fixed DeCE displacement against DeCE-RF feedback-updated controller behavior on the three E4 tasks: cat_crown, tshirt_star, and pillow_vertical_fabric_strip.

## Completion Audit

- Base fixed-vs-feedback rows: $base_rows / 18
- Edit-strength stress metric rows: $stress_rows / 36
- Base summary rows: $base_summary_rows / 2
- Stress summary rows: $stress_summary_rows / 12
- Controller trajectory rows: $trajectory_rows / 18

## Primary Artifacts

- $EXP/e4_controller_base_metrics.csv
- $EXP/e4_edit_strength_metrics.csv
- $OUT/e4_controller_base_summary.csv
- $OUT/e4_edit_strength_summary.csv
- $OUT/e4_controller_trajectory_stats.csv
- $OUT/e4_controller_trajectory_summary.csv
- $OUT/e4_figure5_edit_strength_pareto.png
- $OUT/e4_controller_trajectory_tshirt_star_seed10.png
- $OUT/e4_controller_ablation_summary.md

Interpretation policy: E4 is a contextual controller ablation. It supports a robustness/stabilization claim for feedback-updated clean-estimate control, not the headline E2.2 algorithmic claim by itself.
EOF

find "$EXP" \
  -path "$OUT/*" \
  -type f \
  ! -name "e4_controller_ablation.sha256" \
  | sort > "$OUT/e4_controller_ablation_files.txt"

sha256sum $(cat "$OUT/e4_controller_ablation_files.txt") \
  > "$OUT/e4_controller_ablation.sha256"

sha256sum -c "$OUT/e4_controller_ablation.sha256"

date
