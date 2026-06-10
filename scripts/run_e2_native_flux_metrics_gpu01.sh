#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00
#SBATCH -J e2-flux-metrics
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_native_flux_metrics_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_native_flux_metrics_%j.err

set -u

PROJECT=/cluster/users/grad/2025/25t8103/project
EXP="$PROJECT/experiments/support_v3_2026-06-02"
OUT="$PROJECT/outputs/e2_native_flux_contextual"
PY="$PROJECT/_baselines/envs/fireflow-py310/bin/python"

cd "$PROJECT" || exit 1

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run metrics outside a100-01" >&2
  exit 2
fi

"$PY" scripts/evaluate_paper_metrics.py \
  --outputs-dir "$OUT" \
  --csv-output "$EXP/e2_native_flux_fixed_mask_metrics.csv" \
  --json-output "$EXP/e2_native_flux_fixed_mask_metrics.json" \
  --task-names "cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_vertical_fabric_strip backpack_remove_toy_charm" \
  --method-names "fireflow rf_solver_edit reflex" \
  --seeds "10 11 12" \
  --eval-mask-dir "$EXP/normalized_512/eval_masks" \
  --preserve-floor-csv "$EXP/strict_fixed_mask_metrics.csv"

"$PY" - <<'PY'
import csv
import json
from pathlib import Path

exp = Path("experiments/support_v3_2026-06-02")
metrics_path = exp / "e2_native_flux_fixed_mask_metrics.csv"
out_path = exp / "e2_native_flux_fixed_mask_metrics_with_context.csv"

rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8")))
for row in rows:
    run = row.get("run", "")
    parts = run.split("/")
    if len(parts) >= 3:
        meta_path = Path("outputs/e2_native_flux_contextual") / run / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            row["backbone"] = meta.get("backbone", "FLUX.1-dev")
            row["native_backbone"] = meta.get("native_backbone", "FLUX.1-dev")
            row["resolution"] = "512x512"
            row["normalization_policy"] = meta.get("normalization_policy", "")
            row["e2_layer"] = "E2.3 native implementation context"
            row["claim_boundary"] = "contextual native FLUX row; not E2.2 same-backbone algorithmic evidence"

fieldnames = sorted({key for row in rows for key in row})
with out_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out_path}")
PY
