#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J e2-4-t1t4
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_support_matched_t1_t4_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_support_matched_t1_t4_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
OUT="$PROJECT/outputs/e2_support_matched_diagnostic_t1_t4"
PY="$PROJECT/_baselines/envs/fireflow-py310/bin/python"
TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run E2.4 T1-T4 metrics outside a100-01" >&2
  exit 2
fi

"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$OUT" \
  --csv-output "$EXP/e2_support_matched_t1_t4_fixed_mask_metrics.csv" \
  --json-output "$EXP/e2_support_matched_t1_t4_fixed_mask_metrics.json" \
  --task-names "$TASKS" \
  --method-names "direct_target_raw direct_target_mask_blend flowedit_mask_blend support_v3_controller_rmsgap" \
  --seeds "10 11 12" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --preserve-floor-csv "$EXP/e4_t1_t4_reconstruction_floor_metrics.csv"

"$PY" - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

exp = Path("experiments/support_v3_2026-06-02")
rows = list(csv.DictReader((exp / "e2_support_matched_t1_t4_fixed_mask_metrics.csv").open(newline="", encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
    groups[row["method"]].append(row)

def avg(items, key):
    values = []
    for item in items:
        try:
            if item.get(key, "") != "":
                values.append(float(item[key]))
        except ValueError:
            pass
    return "" if not values else f"{sum(values) / len(values):.4f}"

summary = []
for method in ["direct_target_raw", "direct_target_mask_blend", "flowedit_mask_blend", "support_v3_controller_rmsgap"]:
    items = groups[method]
    summary.append({
        "method": method,
        "n": str(len(items)),
        "outside_mask_l1_mean": avg(items, "outside_mask_l1"),
        "inside_mask_l1_mean": avg(items, "inside_mask_l1"),
        "source_ssim_luma_mean": avg(items, "source_ssim_luma"),
        "edit_score_mean": avg(items, "edit_score"),
    })
out = exp / "e2_support_matched_t1_t4_summary.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
print(f"wrote {out}")
PY
