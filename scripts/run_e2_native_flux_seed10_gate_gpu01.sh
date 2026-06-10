#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=12:00:00
#SBATCH -J e2-flux-gate
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_native_flux_gate/e2_native_flux_seed10_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/e2_native_flux_gate/e2_native_flux_seed10_%j.err

set -u

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
MANIFEST="$PROJECT/experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv"
LOG_DIR="$BASE/logs/e2_native_flux_gate/e2_native_flux_seed10_${SLURM_JOB_ID:-manual}"
SUMMARY="$LOG_DIR/summary.txt"
GPU_LOG="$LOG_DIR/gpu.log"
TASKS="cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_vertical_fabric_strip backpack_remove_toy_charm"

export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR" "$HF_HUB_CACHE"
cd "$PROJECT" || exit 1

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run E2 native FLUX gate outside a100-01" >&2
  exit 2
fi

cp "$MANIFEST" "$MANIFEST.before_e2_native_flux_seed10_${SLURM_JOB_ID:-manual}.bak"
touch "$SUMMARY"

monitor_gpu() {
  while true; do
    echo "==== $(date)"
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv || true
    nvidia-smi || true
    sleep 300
  done >> "$GPU_LOG" 2>&1
}

run_step() {
  local name="$1"
  local log="$LOG_DIR/$name.log"
  shift
  echo "==== $name" > "$log"
  echo "date=$(date)" >> "$log"
  echo "cmd=$*" >> "$log"
  if "$@" >> "$log" 2>&1; then
    echo "$name,ok,$log" >> "$SUMMARY"
    echo "[$name] ok"
  else
    local status=$?
    echo "$name,failed:$status,$log" >> "$SUMMARY"
    echo "[$name] failed:$status"
  fi
}

monitor_gpu &
MON_PID=$!

run_step fireflow_seed10 \
  "$BASE/envs/fireflow-py310/bin/python" \
  scripts/archive_legacy_2026-05-11/run_fireflow_baseline.py \
  --manifest "$MANIFEST" \
  --fireflow-root "$SRC/FireFlow" \
  --python "$BASE/envs/fireflow-py310/bin/python" \
  --tasks "$TASKS" \
  --seeds 10 \
  --max-image-size 512

run_step rf_solver_edit_seed10 \
  "$BASE/envs/rf-solver-edit-py310/bin/python" \
  scripts/archive_legacy_2026-05-11/run_rf_solver_edit_baseline.py \
  --manifest "$MANIFEST" \
  --rf-solver-root "$SRC/RF-Solver-Edit/FLUX_Image_Edit" \
  --python "$BASE/envs/rf-solver-edit-py310/bin/python" \
  --tasks "$TASKS" \
  --seeds 10 \
  --max-image-size 512

run_step reflex_seed10 \
  "$BASE/envs/reflex-py310/bin/python" \
  scripts/archive_legacy_2026-05-11/run_reflex_baseline.py \
  --manifest "$MANIFEST" \
  --reflex-root "$SRC/ReFlex" \
  --python "$BASE/envs/reflex-py310/bin/python" \
  --tasks "$TASKS" \
  --seeds 10 \
  --max-image-size 512

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true

echo "summary=$SUMMARY"
cat "$SUMMARY"

echo "manifest_status_counts:"
"$BASE/envs/fireflow-py310/bin/python" - "$MANIFEST" <<'PY'
import csv
import sys
from collections import Counter
manifest = sys.argv[1]
rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
for baseline in ["fireflow", "rf_solver_edit", "reflex"]:
    subset = [r for r in rows if r["baseline"] == baseline and r["seed"] == "10"]
    print(baseline, dict(Counter(r["status"] for r in subset)))
PY
