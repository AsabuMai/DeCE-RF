#!/usr/bin/env bash
#SBATCH -J t4_vase3
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 02:30:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t4_vase_shape_repair3_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_t4_vase_shape_repair3_seed10_20260606_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
ARTIFACT="${ROOT}/experiments/support_v3_2026-06-02/phase2_t4_vase_shape_repair3_seed10_${SLURM_JOB_ID:-manual}.tgz"

mkdir -p "${RUN_DIR}"

echo "[t4-vase3] start=$(date -Is)"
echo "[t4-vase3] host=$(hostname)"
echo "[t4-vase3] job_id=${SLURM_JOB_ID:-none}"
echo "[t4-vase3] run_dir=${RUN_DIR}"
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

TASKS_ONE="yellow_vase_blue_phase2"
METHODS_ONE="support_v3_controller_rmsgap"
SEEDS_ONE="10"
TASK_OUT="${ROOT}/outputs/pretty_matrix/yellow_vase_blue_phase2/support_v3_controller_rmsgap/seed_10"
BACKUP_DIR="${ROOT}/outputs/pretty_matrix_registry_backup/t4_vase_shape_repair3_seed10_${SLURM_JOB_ID:-manual}"

run_variant() {
  local label="$1"
  local ref="$2"
  local color="$3"
  local boundary_preserve="$4"
  local surf_dilate="$5"
  local manifest="${RUN_DIR}/command_manifest_${label}.txt"
  local summary="${RUN_DIR}/sd3_batch_summary_${label}.json"

  echo "[t4-vase3] variant=${label} ref=${ref} color=${color} bpres=${boundary_preserve} surf_dilate=${surf_dilate}"
  rm -f "${manifest}"

  if [[ -d "${TASK_OUT}" ]]; then
    mkdir -p "${BACKUP_DIR}/${label}"
    rm -rf "${BACKUP_DIR}/${label}/seed_10"
    mv "${TASK_OUT}" "${BACKUP_DIR}/${label}/seed_10"
  fi

  VASE_RECOLOR_SOURCE_COLOR="" \
  VASE_RECOLOR_FILL_HOLES="0" \
  VASE_RECOLOR_SURFACE_REFINE_ERODE_ITERATIONS="0" \
  VASE_RECOLOR_SURFACE_REFINE_DILATE_KERNEL="${surf_dilate}" \
  VASE_RECOLOR_SURFACE_REFINE_DILATE_ITERATIONS="1" \
  VASE_RECOLOR_REF_GUIDANCE_SCALE="${ref}" \
  VASE_RECOLOR_COLOR_GUIDANCE_SCALE="${color}" \
  VASE_RECOLOR_TRAJECTORY_SUBJECT_PRESERVE_SCALE="0.30" \
  VASE_RECOLOR_TRIMAP_BOUNDARY_PRESERVE_SCALE="${boundary_preserve}" \
  BATCH_MANIFEST="${manifest}" \
  PREPARE_ONLY=1 \
  SKIP_EXISTING=0 \
  TASKS="${TASKS_ONE}" \
  METHODS="${METHODS_ONE}" \
  SEEDS="${SEEDS_ONE}" \
  bash scripts/run_pretty_matrix.sh

  echo "[t4-vase3] manifest_count_${label}=$(wc -l < "${manifest}")"

  "${PYTHON}" scripts/run_sd3_batch.py \
    --manifest "${manifest}" \
    --summary-output "${summary}"

  for required in result.png metadata.json stats.json command.txt; do
    test -s "${TASK_OUT}/${required}"
  done

  mkdir -p "${RUN_DIR}/variant_outputs/${label}"
  cp -a "${TASK_OUT}" "${RUN_DIR}/variant_outputs/${label}/seed_10"
}

run_variant sem_c012_r054_b010_d1 0.54 0.12 0.10 1
run_variant sem_c012_r054_b010_d2 0.54 0.12 0.10 2
run_variant sem_c011_r053_b010_d2 0.53 0.11 0.10 2
run_variant sem_c012_r054_b015_d2 0.54 0.12 0.15 2

tar -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  "experiments/support_v3_2026-06-02/${RUN_ID}"

echo "[t4-vase3] artifact=${ARTIFACT}"
echo "[t4-vase3] end=$(date -Is)"
