#!/usr/bin/env bash
#SBATCH -J t4_rim
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 02:30:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t4_office_chair_rim_sweep_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_t4_office_chair_rim_sweep_seed10_20260605_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
ARTIFACT="${ROOT}/experiments/support_v3_2026-06-02/phase2_t4_office_chair_rim_sweep_seed10_${SLURM_JOB_ID:-manual}.tgz"

mkdir -p "${RUN_DIR}"

echo "[t4-rim] start=$(date -Is)"
echo "[t4-rim] host=$(hostname)"
echo "[t4-rim] job_id=${SLURM_JOB_ID:-none}"
echo "[t4-rim] run_dir=${RUN_DIR}"
nvidia-smi || true

export PYTHON="${ROOT}/.venv/bin/python"
export HF_HOME="${ROOT}/../.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MODEL_OFFLOAD=0
export LOW_VRAM=0
export NUM_INFERENCE_STEPS=28
export N_MAX=24
export MAX_IMAGE_SIZE=512
export REGENERATE_MASKS=1
export REUSE_SEMANTIC_MASKS=0

TASKS_ONE="red_office_chair_to_blue_office_chair"
METHODS_ONE="support_v3_controller_rmsgap"
SEEDS_ONE="10"
TASK_OUT="${ROOT}/outputs/pretty_matrix/red_office_chair_to_blue_office_chair/support_v3_controller_rmsgap/seed_10"
BACKUP_DIR="${ROOT}/outputs/pretty_matrix_registry_backup/t4_office_chair_rim_sweep_seed10_${SLURM_JOB_ID:-manual}"

run_variant() {
  local label="$1"
  local dilate="$2"
  local color="$3"
  local ref="$4"
  local manifest="${RUN_DIR}/command_manifest_${label}.txt"
  local summary="${RUN_DIR}/sd3_batch_summary_${label}.json"

  echo "[t4-rim] variant=${label} dilate=${dilate} color=${color} ref=${ref}"
  rm -f "${manifest}"

  if [[ -d "${TASK_OUT}" ]]; then
    mkdir -p "${BACKUP_DIR}/${label}"
    rm -rf "${BACKUP_DIR}/${label}/seed_10"
    mv "${TASK_OUT}" "${BACKUP_DIR}/${label}/seed_10"
  fi

  OFFICE_RECOLOR_SOURCE_DILATE_KERNEL="${dilate}" \
  OFFICE_RECOLOR_SOURCE_DILATE_ITERATIONS="1" \
  OFFICE_RECOLOR_SOURCE_DILATE_MIN_SATURATION="0.12" \
  OFFICE_RECOLOR_COLOR_GUIDANCE_SCALE="${color}" \
  OFFICE_RECOLOR_REF_GUIDANCE_SCALE="${ref}" \
  BATCH_MANIFEST="${manifest}" \
  PREPARE_ONLY=1 \
  SKIP_EXISTING=0 \
  TASKS="${TASKS_ONE}" \
  METHODS="${METHODS_ONE}" \
  SEEDS="${SEEDS_ONE}" \
  bash scripts/run_pretty_matrix.sh

  echo "[t4-rim] manifest_count_${label}=$(wc -l < "${manifest}")"

  "${PYTHON}" scripts/run_sd3_batch.py \
    --manifest "${manifest}" \
    --summary-output "${summary}"

  for required in result.png metadata.json stats.json command.txt; do
    test -s "${TASK_OUT}/${required}"
  done

  mkdir -p "${RUN_DIR}/variant_outputs/${label}"
  cp -a "${TASK_OUT}" "${RUN_DIR}/variant_outputs/${label}/seed_10"
}

run_variant d07_c010_r058 7 0.10 0.58
run_variant d11_c010_r058 11 0.10 0.58
run_variant d11_c014_r062 11 0.14 0.62

tar -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  "experiments/support_v3_2026-06-02/${RUN_ID}"

echo "[t4-rim] artifact=${ARTIFACT}"
echo "[t4-rim] end=$(date -Is)"
