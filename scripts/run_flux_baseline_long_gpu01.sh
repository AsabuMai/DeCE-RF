#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=03:30:00
#SBATCH -J flux-long
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_matrix/flux_long_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_matrix/flux_long_%j.err

set -u

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
SRC="$BASE/src"
OUT_ROOT="$BASE/generation_matrix"
RUN_ID="flux_long_${SLURM_JOB_ID:-manual}"
RUN_DIR="$OUT_ROOT/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
SUMMARY="$RUN_DIR/summary.csv"
SOURCE_IMAGE="${SOURCE_IMAGE:-$PROJECT/data/paper_images/cat_sitting_in_grass_1024.jpg}"
SOURCE_PROMPT="A photo of a cat sitting in grass."
TARGET_PROMPT="A photo of the same cat sitting in the same grass, wearing a small golden crown on its head."
TIMEOUT_PER_BASELINE=90m
HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"

export HF_HUB_CACHE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR" "$HF_HUB_CACHE"
cd "$PROJECT" || exit 1

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run FLUX baselines outside a100-01" >&2
  exit 2
fi

printf 'baseline,status,output,run_log,gpu_log,env_log,note\n' > "$SUMMARY"

csv_escape() {
  python3 - "$1" <<'PY'
import csv
import io
import sys
buf = io.StringIO()
writer = csv.writer(buf)
writer.writerow([sys.argv[1]])
print(buf.getvalue().strip())
PY
}

record() {
  local baseline="$1"
  local status="$2"
  local output="$3"
  local run_log="$4"
  local gpu_log="$5"
  local env_log="$6"
  local note="$7"
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$(csv_escape "$baseline")" \
    "$(csv_escape "$status")" \
    "$(csv_escape "$output")" \
    "$(csv_escape "$run_log")" \
    "$(csv_escape "$gpu_log")" \
    "$(csv_escape "$env_log")" \
    "$(csv_escape "$note")" >> "$SUMMARY"
  echo "[$baseline] $status :: $note"
}

gpu_snapshot() {
  local title="$1"
  echo "==== $title"
  date
  nvidia-smi -L || true
  nvidia-smi || true
  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv || true
}

monitor_gpu() {
  local log="$1"
  while true; do
    gpu_snapshot "periodic"
    sleep 60
  done >> "$log" 2>&1
}

write_env_log() {
  local baseline="$1"
  local python_bin="$2"
  local env_log="$3"
  {
    echo "baseline=$baseline"
    echo "date=$(date)"
    echo "hostname=$(hostname)"
    echo "pwd=$(pwd)"
    echo "python=$python_bin"
    "$python_bin" --version
    "$python_bin" - <<'PY'
import importlib
mods = ["torch", "diffusers", "transformers", "accelerate", "huggingface_hub"]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR:{type(exc).__name__}:{exc}")
try:
    import torch
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_version={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"torch_device_count={torch.cuda.device_count()}")
        print(f"torch_device_0={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"torch_cuda_probe_error={type(exc).__name__}:{exc}")
PY
    echo "HF_HUB_CACHE=$HF_HUB_CACHE"
    echo "SOURCE_IMAGE=$SOURCE_IMAGE"
    "$python_bin" - "$SOURCE_IMAGE" <<'PY'
from PIL import Image
import sys
path = sys.argv[1]
im = Image.open(path)
print(f"source_size={im.size[0]}x{im.size[1]}")
PY
    du -sh "$HF_HUB_CACHE/models--black-forest-labs--FLUX.1-dev" 2>/dev/null || true
    gpu_snapshot "env-start"
  } > "$env_log" 2>&1
}

run_with_monitor() {
  local baseline="$1"
  local workdir="$2"
  local python_bin="$3"
  local run_log="$4"
  local gpu_log="$5"
  shift 5

  echo "==== $baseline" > "$run_log"
  echo "date=$(date)" >> "$run_log"
  echo "workdir=$workdir" >> "$run_log"
  echo "timeout=$TIMEOUT_PER_BASELINE" >> "$run_log"
  gpu_snapshot "before-$baseline" > "$gpu_log" 2>&1

  monitor_gpu "$gpu_log" &
  local mon_pid=$!

  (
    cd "$workdir" || exit 1
    timeout "$TIMEOUT_PER_BASELINE" "$python_bin" "$@"
  ) >> "$run_log" 2>&1
  local status=$?

  kill "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true
  gpu_snapshot "after-$baseline" >> "$gpu_log" 2>&1
  echo "exit_status=$status" >> "$run_log"
  return "$status"
}

run_fireflow() {
  local baseline=FireFlow
  local out_dir="$RUN_DIR/fireflow"
  local run_log="$LOG_DIR/fireflow.run.log"
  local gpu_log="$LOG_DIR/fireflow.gpu.log"
  local env_log="$LOG_DIR/fireflow.env.log"
  local python_bin="$BASE/envs/fireflow-py310/bin/python"
  mkdir -p "$out_dir"
  write_env_log "$baseline" "$python_bin" "$env_log"
  run_with_monitor "$baseline" "$SRC/FireFlow/src" "$python_bin" "$run_log" "$gpu_log" \
    edit.py \
    --source_prompt "$SOURCE_PROMPT" \
    --target_prompt "$TARGET_PROMPT" \
    --guidance 2 \
    --source_img_dir "$SOURCE_IMAGE" \
    --num_steps 2 \
    --inject 1 \
    --start_layer_index 0 \
    --end_layer_index 37 \
    --name flux-dev \
    --sampling_strategy fireflow \
    --output_prefix smoke_fireflow \
    --output_dir "$out_dir" \
    --feature_path "$out_dir/features" \
    --offload \
    --seed 10
  local status=$?
  local result
  result="$(find "$out_dir" -type f \( -name '*.jpg' -o -name '*.png' \) | head -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$run_log" "$gpu_log" "$env_log" "Generated with 90m timeout."
  else
    record "$baseline" "failed_runtime" "${result:-}" "$run_log" "$gpu_log" "$env_log" "Exit status $status with 90m timeout."
  fi
}

run_rf_solver() {
  local baseline=RF-Solver-Edit
  local out_dir="$RUN_DIR/rf_solver_edit"
  local run_log="$LOG_DIR/rf_solver_edit.run.log"
  local gpu_log="$LOG_DIR/rf_solver_edit.gpu.log"
  local env_log="$LOG_DIR/rf_solver_edit.env.log"
  local python_bin="$BASE/envs/rf-solver-edit-py310/bin/python"
  mkdir -p "$out_dir"
  write_env_log "$baseline" "$python_bin" "$env_log"
  run_with_monitor "$baseline" "$SRC/RF-Solver-Edit/FLUX_Image_Edit/src" "$python_bin" "$run_log" "$gpu_log" \
    edit.py \
    --source_prompt "$SOURCE_PROMPT" \
    --target_prompt "$TARGET_PROMPT" \
    --guidance 2 \
    --source_img_dir "$SOURCE_IMAGE" \
    --num_steps 2 \
    --inject 3 \
    --name flux-dev \
    --output_dir "$out_dir" \
    --feature_path "$out_dir/features" \
    --offload
  local status=$?
  local result
  result="$(find "$out_dir" -type f \( -name '*.jpg' -o -name '*.png' \) | head -1)"
  if [[ "$status" -eq 0 && -n "$result" ]]; then
    record "$baseline" "complete" "$result" "$run_log" "$gpu_log" "$env_log" "Generated with 90m timeout."
  else
    record "$baseline" "failed_runtime" "${result:-}" "$run_log" "$gpu_log" "$env_log" "Exit status $status with 90m timeout."
  fi
}

run_fireflow
run_rf_solver

echo "summary=$SUMMARY"
cat "$SUMMARY"
