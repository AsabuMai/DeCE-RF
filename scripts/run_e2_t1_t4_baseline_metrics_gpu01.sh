#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J e2-t1t4-metrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_t1_t4_baseline_metrics_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_t1_t4_baseline_metrics_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
PY="$PROJECT/_baselines/envs/fireflow-py310/bin/python"
TASKS="cat_crown dog_bow_tie_phase2 dog_front_sunglasses_phase2 bowl_apple_inside white_bowl_orange_tabletop_phase2 brown_bowl_lemon_phase2 tshirt_star mug_heart tote_leaf red_office_chair_to_blue_office_chair green_mug_orange_phase2 yellow_vase_blue_phase2"

cd "$PROJECT"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run E2 T1-T4 metrics outside a100-01" >&2
  exit 2
fi

"$PY" scripts/prepare_e2_t1_t4_baseline_matrix.py

"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$PROJECT/outputs/e2_t1_t4_baseline_matrix" \
  --csv-output "$EXP/e2_t1_t4_baseline_fixed_mask_metrics.csv" \
  --json-output "$EXP/e2_t1_t4_baseline_fixed_mask_metrics.json" \
  --task-names "$TASKS" \
  --method-names "flowedit flowalign splitflow fireflow rf_solver_edit reflex" \
  --seeds "10 11 12" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --preserve-floor-csv "$EXP/e4_t1_t4_reconstruction_floor_metrics.csv" \
  --clip-model openai/clip-vit-large-patch14 \
  --dino-model facebook/dinov2-base \
  --allow-download

"$PY" - <<'PY'
import csv
from collections import Counter, defaultdict
from pathlib import Path

exp = Path("experiments/support_v3_2026-06-02")
metrics = list(csv.DictReader((exp / "e2_t1_t4_baseline_fixed_mask_metrics.csv").open(newline="", encoding="utf-8")))
manifest = list(csv.DictReader((exp / "e2_t1_t4_formal_baseline_manifest.csv").open(newline="", encoding="utf-8")))
summary = []
status_by_baseline = defaultdict(Counter)
for row in manifest:
    status_by_baseline[row["baseline"]][row["status"]] += 1
metrics_by_method = defaultdict(list)
for row in metrics:
    metrics_by_method[row["method"]].append(row)

def avg(rows, key):
    values = []
    for row in rows:
        try:
            if row.get(key, "") != "":
                values.append(float(row[key]))
        except ValueError:
            pass
    return "" if not values else f"{sum(values) / len(values):.4f}"

for baseline in ["flowedit", "flowalign", "splitflow", "fireflow", "rf_solver_edit", "reflex"]:
    rows = metrics_by_method[baseline]
    counts = status_by_baseline[baseline]
    summary.append({
        "baseline": baseline,
        "manifest_complete": str(counts.get("complete", 0)),
        "manifest_failed": str(counts.get("failed", 0)),
        "metric_rows": str(len(rows)),
        "outside_mask_l1_mean": avg(rows, "outside_mask_l1"),
        "inside_mask_l1_mean": avg(rows, "inside_mask_l1"),
        "source_ssim_luma_mean": avg(rows, "source_ssim_luma"),
        "edit_score_mean": avg(rows, "edit_score"),
    })
out = exp / "e2_t1_t4_baseline_summary.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
print(f"wrote {out}")
PY
