#!/usr/bin/env bash
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00
#SBATCH -J gen-smoke-new
#SBATCH -o /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_smoke/new_baselines_%j.out
#SBATCH -e /cluster/users/grad/2025/25t8103/project/_baselines/logs/generation_smoke/new_baselines_%j.err

set -euo pipefail

PROJECT=/cluster/users/grad/2025/25t8103/project
BASE="$PROJECT/_baselines"
CONFIG_DIR="$BASE/run_smoke_configs"
LOG_DIR="$BASE/logs/generation_smoke"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$PROJECT/.cache/huggingface/hub}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT"
mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$HF_HUB_CACHE"

host="$(hostname)"
echo "host=$host"
if [[ "$host" != "a100-01.gpu01.cis.k.hosei.ac.jp" ]]; then
  echo "Refusing to run generation smoke outside a100-01" >&2
  exit 2
fi

cat > "$CONFIG_DIR/otrf_smoke_dataset.yaml" <<YAML
-
  input_img: "$BASE/src/OT-RF/assets/face_input.jpg"
  source_prompt: "face of a person"
  target_prompts:
    - "face of a person wearing glasses"
YAML

cat > "$CONFIG_DIR/otrf_smoke_exp.yaml" <<YAML
-
  exp_name: "smoke_otrf_sd3"
  model_type: "SD3"
  T_steps: 4
  n_avg: 1
  src_guidance_scale: 3.5
  tar_guidance_scale: 7.5
  n_min: 0
  n_max: 2
  seed: 42
  dataset_yaml: "$CONFIG_DIR/otrf_smoke_dataset.yaml"
YAML

cat > "$CONFIG_DIR/dvrf_smoke_dataset.yaml" <<YAML
-
  input_img: "$BASE/src/DeltaRectifiedFlowSampling/images/a_cat_sitting_on_a_table.png"
  source_prompt: "A cat sitting on a table."
  target_prompts:
    - "A lion sitting on a table."
  target_codes:
    - "Lion"
YAML

cat > "$CONFIG_DIR/dvrf_smoke_exp.yaml" <<YAML
-
  exp_name: "smoke_dvrf_sd3"
  dataset_yaml: "$CONFIG_DIR/dvrf_smoke_dataset.yaml"
  model_type: "SD3"
  T_steps: 4
  B: 1
  src_guidance_scale: 3.5
  tgt_guidance_scale: 7.5
  num_steps: 1
  seed: 41
  eta: 1.0
  scheduler_strategy: "descending"
  lr: 0.01
  optimizer: "SGD"
YAML

run_case() {
  local name="$1"
  local cwd="$2"
  shift 2
  echo "==== $name"
  echo "cwd=$cwd"
  echo "cmd=$*"
  (
    cd "$cwd"
    timeout 45m "$@"
  )
  local status=$?
  echo "status=$status"
  return "$status"
}

set +e
run_case \
  "OT-RF generation smoke" \
  "$BASE/src/OT-RF" \
  "$BASE/envs/ot-rf-otip-py310/bin/python" \
  run_script.py \
  --exp_yaml "$CONFIG_DIR/otrf_smoke_exp.yaml" \
  --device_number 0
otrf_status=$?

run_case \
  "DVRF generation smoke" \
  "$BASE/src/DeltaRectifiedFlowSampling" \
  "$BASE/envs/dvrf-py310/bin/python" \
  edit.py \
  --exp_yaml "$CONFIG_DIR/dvrf_smoke_exp.yaml" \
  --device_number 0
dvrf_status=$?
set -e

echo "otrf_status=$otrf_status"
echo "dvrf_status=$dvrf_status"

if [[ "$otrf_status" -ne 0 || "$dvrf_status" -ne 0 ]]; then
  exit 1
fi

echo generation_smoke_complete
