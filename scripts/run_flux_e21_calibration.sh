#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ROOT:-}" ]]; then
  ROOT="${ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="${SLURM_SUBMIT_DIR}"
else
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
RUN_ID="${RUN_ID:-flux_e21_calibration_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/${RUN_ID}}"
TASKS="${TASKS:-cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_vertical_fabric_strip backpack_remove_toy_charm}"
METHODS="${METHODS:-base_only_flux direct_target_flux}"
SEEDS="${SEEDS:-10 11}"
EVAL_MASK_DIR="${EVAL_MASK_DIR:-${ROOT}/experiments/support_v3_2026-06-02/eval_masks}"
FLUX_MODEL_ID="${FLUX_MODEL_ID:-black-forest-labs/FLUX.1-dev}"
HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
FLUX_LOCAL_FILES_ONLY="${FLUX_LOCAL_FILES_ONLY:-1}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-512}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"
N_MAX="${N_MAX:-24}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
FLUX_MODEL_OFFLOAD="${FLUX_MODEL_OFFLOAD:-0}"

host="$(hostname -f 2>/dev/null || hostname)"
if [[ "${ALLOW_NON_A100:-${ALLOW_NON_A100_02:-0}}" != "1" && "${host}" != a100-* ]]; then
  echo "[flux-e21] refusing to run on ${host}; use a Slurm A100 allocation or set ALLOW_NON_A100=1" >&2
  exit 2
fi

source "${ROOT}/scripts/core6_tasks.sh"
export HF_HOME
export HUGGINGFACE_HUB_CACHE

local_files_args=()
if [[ "${FLUX_LOCAL_FILES_ONLY}" == "1" ]]; then
  local_files_args+=("--local-files-only")
fi
if [[ "${FLUX_MODEL_OFFLOAD}" == "1" ]]; then
  local_files_args+=("--model-offload")
fi

mkdir -p "${OUTPUT_ROOT}"
echo "[flux-e21] host=${host}"
echo "[flux-e21] output_root=${OUTPUT_ROOT}"
echo "[flux-e21] tasks=${TASKS}"
echo "[flux-e21] methods=${METHODS}"
echo "[flux-e21] seeds=${SEEDS}"
echo "[flux-e21] eval_mask_dir=${EVAL_MASK_DIR}"
echo "[flux-e21] hf_home=${HF_HOME}"
echo "[flux-e21] hub_cache=${HUGGINGFACE_HUB_CACHE}"
echo "[flux-e21] local_files_only=${FLUX_LOCAL_FILES_ONLY}"
echo "[flux-e21] model_offload=${FLUX_MODEL_OFFLOAD}"

for task in ${TASKS}; do
  task_config "${task}"
  for method in ${METHODS}; do
    for seed in ${SEEDS}; do
      out_dir="${OUTPUT_ROOT}/${TASK_NAME}/${method}/seed_${seed}"
      out_image="${out_dir}/image.png"
      stats_json="${out_dir}/stats.json"
      metadata_json="${out_dir}/metadata.json"
      if [[ "${SKIP_EXISTING:-1}" == "1" && -s "${out_image}" && -s "${stats_json}" && -s "${metadata_json}" ]]; then
        echo "[flux-e21] skip ${TASK_NAME} ${method} seed=${seed}"
        continue
      fi
      mkdir -p "${out_dir}"
      echo "[flux-e21] run ${TASK_NAME} ${method} seed=${seed}"
      support_args=()
      if [[ "${method}" == "dece_rf_flux" ]]; then
        eval_mask="${EVAL_MASK_DIR}/${TASK_NAME}_eval_mask.png"
        if [[ ! -s "${eval_mask}" ]]; then
          echo "[flux-e21] missing eval mask for ${TASK_NAME}: ${eval_mask}" >&2
          exit 2
        fi
        support_args+=(
          --semantic-base-mask "${eval_mask}"
          --support-control-mode "${FLUX_SUPPORT_CONTROL_MODE:-operation}"
          --edit-operation "${SUPPORT_EDIT_OPERATION}"
          --support-relation "${SUPPORT_V3_RELATION}"
          --support-candidate "${FLUX_DECE_SUPPORT_CANDIDATE:-${SUPPORT_V3_CANDIDATE}}"
          --support-external-mask-role "${FLUX_DECE_EXTERNAL_MASK_ROLE:-attention}"
          --rec-guidance-scale "${FLUX_DECE_REC_GUIDANCE_SCALE:-0.22}"
          --edit-hedit-guidance-scale "${FLUX_DECE_HEDIT_GUIDANCE_SCALE:-0.65}"
          --edit-guidance-scale "${FLUX_DECE_EDIT_GUIDANCE_SCALE:-0.0}"
          --edit-region-guidance-scale "${FLUX_DECE_REGION_GUIDANCE_SCALE:-0.0}"
          --edit-local-target-guidance-scale "${FLUX_DECE_LOCAL_TARGET_GUIDANCE_SCALE:-0.0}"
          --alpha-schedule "${FLUX_DECE_ALPHA_SCHEDULE:-constant}"
          --beta-schedule "${FLUX_DECE_BETA_SCHEDULE:-constant}"
          --linear-path-t-min "${FLUX_DECE_LINEAR_PATH_T_MIN:-0.05}"
          --rec-stop-timestep "${FLUX_DECE_REC_STOP_TIMESTEP:-0.08}"
          --trajectory-preserve-scale "${FLUX_DECE_TRAJECTORY_PRESERVE_SCALE:-0.12}"
          --trajectory-subject-preserve-scale "${FLUX_DECE_TRAJECTORY_SUBJECT_PRESERVE_SCALE:-0.0}"
          --region-target-transport-scale "${FLUX_DECE_REGION_TARGET_TRANSPORT_SCALE:-0.0}"
          --region-target-outside-lock-scale "${FLUX_DECE_REGION_TARGET_OUTSIDE_LOCK_SCALE:-0.0}"
          --mask-output-dir "${out_dir}/masks"
        )
        if [[ "${FLUX_DECE_ADAPTIVE_CLEAN_CONTROL:-1}" == "1" ]]; then
          support_args+=(
            --adaptive-clean-control
            --adaptive-rmsgap-mode "${FLUX_DECE_ADAPTIVE_RMSGAP_MODE:-legacy}"
          )
        fi
        if [[ "${SUPPORT_EDIT_OPERATION}" == "recolor" ]]; then
          support_args+=(
            --recolor-target "${FLUX_RECOLOR_TARGET:-blue}"
            --recolor-clean-projection-scale "${FLUX_RECOLOR_CLEAN_PROJECTION_SCALE:-0.35}"
          )
        fi
        if [[ "${SUPPORT_EDIT_OPERATION}" == "remove_object" ]]; then
          support_args+=(
            --removal-controller-mode clean_fill
            --removal-fill-scale "${FLUX_REMOVAL_FILL_SCALE:-0.70}"
            --removal-suppression-scale "${FLUX_REMOVAL_SUPPRESSION_SCALE:-0.35}"
            --removal-ring-rec-scale "${FLUX_REMOVAL_RING_REC_SCALE:-0.40}"
          )
        fi
        if [[ "${FLUX_USE_ATTENTION_SUPPORT:-0}" == "1" ]]; then
          support_args+=(--use-flux-attention-support)
          if [[ -n "${SUPPORT_NEW_TOKENS}" ]]; then
            support_args+=(--new-tokens "${SUPPORT_NEW_TOKENS}")
          fi
          if [[ -n "${SUPPORT_HOST_TOKENS}" ]]; then
            support_args+=(--host-tokens "${SUPPORT_HOST_TOKENS}")
          fi
          support_args+=(--flux-attention-layer-stride "${FLUX_ATTENTION_LAYER_STRIDE:-2}")
        fi
        local_target_scale="${FLUX_DECE_LOCAL_TARGET_GUIDANCE_SCALE:-0.0}"
        if [[ -n "${SUPPORT_LOCAL_TARGET_PROMPT}" && "${local_target_scale}" != "0" && "${local_target_scale}" != "0.0" ]]; then
          support_args+=(--edit-local-target-prompt "${SUPPORT_LOCAL_TARGET_PROMPT}")
        fi
      fi
      "${PYTHON}" "${ROOT}/run_edit_flux.py" \
        --model-id "${FLUX_MODEL_ID}" \
        --cache-dir "${HUGGINGFACE_HUB_CACHE}" \
        "${local_files_args[@]}" \
        --image "${IMAGE}" \
        --source-prompt "${SOURCE_PROMPT}" \
        --prompt "${TARGET_PROMPT}" \
        --method "${method}" \
        --seed "${seed}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --n-max "${N_MAX}" \
        --max-image-size "${MAX_IMAGE_SIZE}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        "${support_args[@]}" \
        --output "${out_image}" \
        --stats-output "${stats_json}" \
        --metadata-output "${metadata_json}"
    done
  done
done

echo "[flux-e21] done ${OUTPUT_ROOT}"
