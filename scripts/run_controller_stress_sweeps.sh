#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
GPU_DEVICES="${GPU_DEVICES:-${DEVICE}}"
PARALLEL="${PARALLEL:-0}"

TASKS="${TASKS:-P1 P2 P3}"
METHODS="${METHODS:-M17 M18}"
SEEDS="${SEEDS:-10 11 12}"
ANALYSIS_TASKS="${ANALYSIS_TASKS:-cat_crown dog_sunglasses mug_heart}"

EDIT_SCALES="${EDIT_SCALES:-0.5 0.75 1.0 1.25 1.5 2.0}"
PERTURBATIONS="${PERTURBATIONS:-erode dilate shift boundary_noise holes}"

RUN_EDIT_STRENGTH="${RUN_EDIT_STRENGTH:-1}"
RUN_SUPPORT_PERTURB="${RUN_SUPPORT_PERTURB:-1}"
RUN_ANALYZE="${RUN_ANALYZE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REGENERATE_MASKS="${REGENERATE_MASKS:-0}"
ALLOW_MASK_DOWNLOAD="${ALLOW_MASK_DOWNLOAD:-0}"
DRY_RUN="${DRY_RUN:-0}"

ANALYSIS_DIR="${ANALYSIS_DIR:-${ROOT}/experiments/support_v3_2026-05-11/controller_stress}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
ALLOW_METRIC_DOWNLOAD="${ALLOW_METRIC_DOWNLOAD:-0}"
COMPUTE_LPIPS="${COMPUTE_LPIPS:-0}"
LOG_DIR="${LOG_DIR:-${ANALYSIS_DIR}/logs}"

read -r -a GPU_LIST <<< "${GPU_DEVICES}"
DEFAULT_PARALLEL_JOBS="${#GPU_LIST[@]}"
if (( DEFAULT_PARALLEL_JOBS > 2 )); then
  DEFAULT_PARALLEL_JOBS=2
fi
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-${DEFAULT_PARALLEL_JOBS}}"
CLIP_DEVICE="${CLIP_DEVICE:-cuda:${GPU_LIST[0]}}"
JOB_COUNT=0
JOB_FAILED=0
FREE_SLOT=0
GPU_PIDS=()
for _gpu in "${GPU_LIST[@]}"; do
  GPU_PIDS+=("")
done

scale_tag() {
  "${PYTHON}" - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(f"{int(round(value * 100)):03d}")
PY
}

run_matrix() {
  local suffix="$1"
  shift
  echo "[stress] suffix=${suffix} $*"
  env \
    ROOT="${ROOT}" \
    PYTHON="${PYTHON}" \
    DEVICE="${DEVICE}" \
    TASKS="${TASKS}" \
    METHODS="${METHODS}" \
    SEEDS="${SEEDS}" \
    METHOD_NAME_SUFFIX="${suffix}" \
    SKIP_EXISTING="${SKIP_EXISTING}" \
    REGENERATE_MASKS="${REGENERATE_MASKS}" \
    ALLOW_MASK_DOWNLOAD="${ALLOW_MASK_DOWNLOAD}" \
    DRY_RUN="${DRY_RUN}" \
    "$@" \
    "${ROOT}/scripts/run_pretty_matrix.sh"
}

wait_for_slot() {
  while [[ "${PARALLEL}" == "1" ]]; do
    local limit="${MAX_PARALLEL_JOBS}"
    if (( limit > ${#GPU_LIST[@]} )); then
      limit="${#GPU_LIST[@]}"
    fi
    for ((idx = 0; idx < limit; idx++)); do
      local pid="${GPU_PIDS[$idx]}"
      if [[ -z "${pid}" ]]; then
        FREE_SLOT="${idx}"
        return 0
      fi
      if ! kill -0 "${pid}" 2>/dev/null; then
        if ! wait "${pid}"; then
          JOB_FAILED=1
        fi
        GPU_PIDS[$idx]=""
        FREE_SLOT="${idx}"
        return 0
      fi
    done
    sleep 2
  done
}

wait_for_all() {
  if [[ "${PARALLEL}" == "1" ]]; then
    for pid in "${GPU_PIDS[@]}"; do
      if [[ -n "${pid}" ]]; then
        if ! wait "${pid}"; then
          JOB_FAILED=1
        fi
      fi
    done
    if [[ "${JOB_FAILED}" != "0" ]]; then
      echo "[stress] at least one parallel job failed; see ${LOG_DIR}" >&2
      exit 1
    fi
  fi
}

submit_job() {
  local suffix="$1"
  local task="$2"
  local method="$3"
  local seed="$4"
  shift 4
  if [[ "${PARALLEL}" != "1" ]]; then
    echo "[stress] run task=${task} method=${method} seed=${seed} suffix=${suffix} $*"
    env \
      ROOT="${ROOT}" \
      PYTHON="${PYTHON}" \
      DEVICE="${DEVICE}" \
      TASKS="${task}" \
      METHODS="${method}" \
      SEEDS="${seed}" \
      METHOD_NAME_SUFFIX="${suffix}" \
      SKIP_EXISTING="${SKIP_EXISTING}" \
      REGENERATE_MASKS="${REGENERATE_MASKS}" \
      ALLOW_MASK_DOWNLOAD="${ALLOW_MASK_DOWNLOAD}" \
      DRY_RUN="${DRY_RUN}" \
      "$@" \
      "${ROOT}/scripts/run_pretty_matrix.sh"
    return 0
  fi

  wait_for_slot
  local slot="${FREE_SLOT}"
  mkdir -p "${LOG_DIR}"
  local gpu="${GPU_LIST[$slot]}"
  local safe_suffix="${suffix#_}"
  local log="${LOG_DIR}/${task}_${method}_seed${seed}_${safe_suffix}.log"
  echo "[stress] submit slot=${slot} gpu=${gpu} task=${task} method=${method} seed=${seed} suffix=${suffix} log=${log}"
  (
    env \
      ROOT="${ROOT}" \
      PYTHON="${PYTHON}" \
      DEVICE="${gpu}" \
      TASKS="${task}" \
      METHODS="${method}" \
      SEEDS="${seed}" \
      METHOD_NAME_SUFFIX="${suffix}" \
      SKIP_EXISTING="${SKIP_EXISTING}" \
      REGENERATE_MASKS="${REGENERATE_MASKS}" \
      ALLOW_MASK_DOWNLOAD="${ALLOW_MASK_DOWNLOAD}" \
      DRY_RUN="${DRY_RUN}" \
      "$@" \
      "${ROOT}/scripts/run_pretty_matrix.sh"
  ) >"${log}" 2>&1 &
  GPU_PIDS[$slot]="$!"
  JOB_COUNT=$((JOB_COUNT + 1))
}

if [[ "${RUN_EDIT_STRENGTH}" == "1" ]]; then
  for scale in ${EDIT_SCALES}; do
    tag="$(scale_tag "${scale}")"
    for task in ${TASKS}; do
      for method in ${METHODS}; do
        for seed in ${SEEDS}; do
          submit_job "_edit${tag}" "${task}" "${method}" "${seed}" EDIT_STRENGTH_MULTIPLIER="${scale}"
        done
      done
    done
  done
fi

if [[ "${RUN_SUPPORT_PERTURB}" == "1" ]]; then
  for perturb in ${PERTURBATIONS}; do
    case "${perturb}" in
      erode)
        perturb_args=(EDIT_MASK_ERODE_KERNEL="${PERTURB_ERODE_KERNEL:-3}")
        ;;
      dilate)
        perturb_args=(EDIT_MASK_DILATE_KERNEL="${PERTURB_DILATE_KERNEL:-5}")
        ;;
      shift)
        perturb_args=(
          EDIT_MASK_SHIFT_X="${PERTURB_SHIFT_X:-0.04}" \
          EDIT_MASK_SHIFT_Y="${PERTURB_SHIFT_Y:-0.00}"
        )
        ;;
      boundary_noise)
        perturb_args=(
          EDIT_MASK_BOUNDARY_NOISE_SCALE="${PERTURB_BOUNDARY_NOISE_SCALE:-0.65}"
        )
        ;;
      holes)
        perturb_args=(EDIT_MASK_HOLE_FRACTION="${PERTURB_HOLE_FRACTION:-0.18}")
        ;;
      *)
        echo "Unknown perturbation '${perturb}'." >&2
        exit 2
        ;;
    esac
    for task in ${TASKS}; do
      for method in ${METHODS}; do
        for seed in ${SEEDS}; do
          submit_job "_pert_${perturb}" "${task}" "${method}" "${seed}" "${perturb_args[@]}"
        done
      done
    done
  done
fi

wait_for_all

if [[ "${RUN_ANALYZE}" == "1" ]]; then
  analyze_cmd=(
    "${PYTHON}" "${ROOT}/scripts/analyze_controller_stress.py"
    --outputs-dir "${ROOT}/outputs/pretty_matrix"
    --output-dir "${ANALYSIS_DIR}"
    --tasks "${ANALYSIS_TASKS}"
    --seeds "${SEEDS}"
    --edit-scales "${EDIT_SCALES}"
    --perturbations "${PERTURBATIONS}"
    --clip-model "${CLIP_MODEL}"
    --clip-device "${CLIP_DEVICE}"
  )
  if [[ "${ALLOW_METRIC_DOWNLOAD}" == "1" ]]; then
    analyze_cmd+=(--allow-download)
  fi
  if [[ "${COMPUTE_LPIPS}" == "1" ]]; then
    analyze_cmd+=(--compute-lpips)
  fi
  echo "[stress] analyzing into ${ANALYSIS_DIR}"
  "${analyze_cmd[@]}"
fi
