#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"

IMAGE="${IMAGE:-${ROOT}/h_edit_compare/panda.jpg}"
SOURCE_PROMPT="${SOURCE_PROMPT:-A panda is walking in a forest.}"
TARGET_PROMPT="${TARGET_PROMPT:-A panda wearing small black sunglasses is walking in a forest.}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/ode_decoupled_edit}"

EDIT_FIELD_MODE="${EDIT_FIELD_MODE:-rf_text_diff}"
OBJECT_MASK_PROVIDER="${OBJECT_MASK_PROVIDER:-attention_velocity}"
MASK_LAYERING_MODE="${MASK_LAYERING_MODE:-object_contact}"
FINAL_EDIT_MASK="${FINAL_EDIT_MASK:-${EXTERNAL_EDIT_MASK:-}}"
FINAL_EDIT_MASK_MODE="${FINAL_EDIT_MASK_MODE:-replace}"
SUPPORT_MASK="${SUPPORT_MASK:-${SEMANTIC_BASE_MASK:-}}"
ATTENTION_VELOCITY_SUPPORT_PAD_X="${ATTENTION_VELOCITY_SUPPORT_PAD_X:-0.015}"
ATTENTION_VELOCITY_SUPPORT_PAD_Y="${ATTENTION_VELOCITY_SUPPORT_PAD_Y:-0.010}"
ATTENTION_VELOCITY_SUPPORT_MIN_WIDTH="${ATTENTION_VELOCITY_SUPPORT_MIN_WIDTH:-0.18}"
ATTENTION_VELOCITY_SUPPORT_MIN_HEIGHT="${ATTENTION_VELOCITY_SUPPORT_MIN_HEIGHT:-0.065}"

SRC_GUIDANCE_SCALE="${SRC_GUIDANCE_SCALE:-1.0}"
BASE_GUIDANCE_SCALE="${BASE_GUIDANCE_SCALE:-1.0}"
TAR_GUIDANCE_SCALE="${TAR_GUIDANCE_SCALE:-10.5}"
EDIT_HEDIT_GUIDANCE_SCALE="${EDIT_HEDIT_GUIDANCE_SCALE:-0.55}"
EDIT_TEXT_GUIDANCE_SCALE="${EDIT_TEXT_GUIDANCE_SCALE:-0.08}"
EDIT_TEXT_SOURCE_SCALE="${EDIT_TEXT_SOURCE_SCALE:-0.8}"
EDIT_COLOR_GUIDANCE_SCALE="${EDIT_COLOR_GUIDANCE_SCALE:-0.0}"
EDIT_COLOR_SOURCE="${EDIT_COLOR_SOURCE:-}"
EDIT_COLOR_TARGET="${EDIT_COLOR_TARGET:-}"
EDIT_COLOR_MASK_IMAGE="${EDIT_COLOR_MASK_IMAGE:-}"
EDIT_COLOR_MASK_THRESHOLD="${EDIT_COLOR_MASK_THRESHOLD:-0.38}"
EDIT_COLOR_MASK_SOFTNESS="${EDIT_COLOR_MASK_SOFTNESS:-0.10}"
EDIT_COLOR_LUMA_GATE_MIN="${EDIT_COLOR_LUMA_GATE_MIN:-0.0}"
EDIT_COLOR_LUMA_GATE_SOFTNESS="${EDIT_COLOR_LUMA_GATE_SOFTNESS:-0.08}"
EDIT_COLOR_DETAIL_PROTECT_SCALE="${EDIT_COLOR_DETAIL_PROTECT_SCALE:-0.0}"
EDIT_COLOR_DETAIL_PROTECT_THRESHOLD="${EDIT_COLOR_DETAIL_PROTECT_THRESHOLD:-0.35}"
EDIT_COLOR_DETAIL_PROTECT_SOFTNESS="${EDIT_COLOR_DETAIL_PROTECT_SOFTNESS:-0.08}"
EDIT_COLOR_TARGET_CHROMA_SCALE="${EDIT_COLOR_TARGET_CHROMA_SCALE:-1.0}"
EDIT_COLOR_SMOOTH_KERNEL="${EDIT_COLOR_SMOOTH_KERNEL:-5}"
EDIT_COLOR_LUMA_PRESERVE_SCALE="${EDIT_COLOR_LUMA_PRESERVE_SCALE:-0.35}"
EDIT_COLOR_LUMA_GRADIENT_PRESERVE_SCALE="${EDIT_COLOR_LUMA_GRADIENT_PRESERVE_SCALE:-0.15}"
EDIT_REF_GUIDANCE_SCALE="${EDIT_REF_GUIDANCE_SCALE:-0.0}"
EDIT_REF_IMAGE="${EDIT_REF_IMAGE:-}"
EDIT_REF_MASK="${EDIT_REF_MASK:-}"
EDIT_REF_STRUCTURE_IMAGE="${EDIT_REF_STRUCTURE_IMAGE:-}"
EDIT_REF_CHROMA_MODE="${EDIT_REF_CHROMA_MODE:-yuv}"
EDIT_REF_CHROMA_MAGNITUDE_SCALE="${EDIT_REF_CHROMA_MAGNITUDE_SCALE:-1.0}"
EDIT_REF_LUMA_PRESERVE_SCALE="${EDIT_REF_LUMA_PRESERVE_SCALE:-0.35}"
EDIT_REF_GRADIENT_PRESERVE_SCALE="${EDIT_REF_GRADIENT_PRESERVE_SCALE:-0.15}"
EDIT_REF_DARKNESS_GUARD_SCALE="${EDIT_REF_DARKNESS_GUARD_SCALE:-0.0}"
EDIT_REF_DARKNESS_GUARD_MARGIN="${EDIT_REF_DARKNESS_GUARD_MARGIN:-0.03}"
EDIT_REF_SMOOTH_KERNEL="${EDIT_REF_SMOOTH_KERNEL:-1}"
EDIT_REF_LOWFREQ_SUPPRESS_KERNEL="${EDIT_REF_LOWFREQ_SUPPRESS_KERNEL:-0}"
EDIT_REF_LOWFREQ_SUPPRESS_SCALE="${EDIT_REF_LOWFREQ_SUPPRESS_SCALE:-0.0}"
EDIT_REF_SCHEDULE_START="${EDIT_REF_SCHEDULE_START:-0.0}"
EDIT_REF_SCHEDULE_STOP="${EDIT_REF_SCHEDULE_STOP:-0.0}"
EDIT_REF_SCHEDULE_POWER="${EDIT_REF_SCHEDULE_POWER:-1.0}"
EDIT_REF_MAX_STRUCT_RMS_RATIO="${EDIT_REF_MAX_STRUCT_RMS_RATIO:-0.0}"
EDIT_REF_PROJECT_STRUCT_CONFLICT="${EDIT_REF_PROJECT_STRUCT_CONFLICT:-0.0}"
REC_GUIDANCE_SCALE="${REC_GUIDANCE_SCALE:-0.25}"
TRAJECTORY_PRESERVE_SCALE="${TRAJECTORY_PRESERVE_SCALE:-0.15}"
PHOTO_PROMPT_MODE="${PHOTO_PROMPT_MODE:-both}"

mkdir -p "${OUT_DIR}/masks"

case "${OBJECT_MASK_PROVIDER}" in
  semantic|semantic_velocity)
    ;;
  *)
    if [[ -n "${SUPPORT_MASK}" ]]; then
      echo "ERROR: SUPPORT_MASK/SEMANTIC_BASE_MASK only affects OBJECT_MASK_PROVIDER=semantic or semantic_velocity." >&2
      echo "Use FINAL_EDIT_MASK for a late final-M_edit override, or set OBJECT_MASK_PROVIDER=semantic_velocity." >&2
      exit 2
    fi
    ;;
esac

cmd=(
  "${PYTHON}" "${ROOT}/run_edit_sd3.py"
  --image "${IMAGE}"
  --source-prompt "${SOURCE_PROMPT}"
  --prompt "${TARGET_PROMPT}"
  --output "${OUT_DIR}/result.png"
  --stats-output "${OUT_DIR}/stats.json"
  --metadata-output "${OUT_DIR}/metadata.json"
  --mask-output-dir "${OUT_DIR}/masks"
  --edit-field-mode "${EDIT_FIELD_MODE}"
  --object-mask-provider "${OBJECT_MASK_PROVIDER}"
  --attention-velocity-support-pad-x "${ATTENTION_VELOCITY_SUPPORT_PAD_X}"
  --attention-velocity-support-pad-y "${ATTENTION_VELOCITY_SUPPORT_PAD_Y}"
  --attention-velocity-support-min-width "${ATTENTION_VELOCITY_SUPPORT_MIN_WIDTH}"
  --attention-velocity-support-min-height "${ATTENTION_VELOCITY_SUPPORT_MIN_HEIGHT}"
  --mask-layering-mode "${MASK_LAYERING_MODE}"
  --attention-mask-target-words ""
  --src-guidance-scale "${SRC_GUIDANCE_SCALE}"
  --base-guidance-scale "${BASE_GUIDANCE_SCALE}"
  --tar-guidance-scale "${TAR_GUIDANCE_SCALE}"
  --edit-hedit-guidance-scale "${EDIT_HEDIT_GUIDANCE_SCALE}"
  --edit-text-guidance-scale "${EDIT_TEXT_GUIDANCE_SCALE}"
  --edit-text-source-scale "${EDIT_TEXT_SOURCE_SCALE}"
  --edit-color-guidance-scale "${EDIT_COLOR_GUIDANCE_SCALE}"
  --edit-color-mask-image "${EDIT_COLOR_MASK_IMAGE}"
  --edit-color-mask-threshold "${EDIT_COLOR_MASK_THRESHOLD}"
  --edit-color-mask-softness "${EDIT_COLOR_MASK_SOFTNESS}"
  --edit-color-luma-gate-min "${EDIT_COLOR_LUMA_GATE_MIN}"
  --edit-color-luma-gate-softness "${EDIT_COLOR_LUMA_GATE_SOFTNESS}"
  --edit-color-detail-protect-scale "${EDIT_COLOR_DETAIL_PROTECT_SCALE}"
  --edit-color-detail-protect-threshold "${EDIT_COLOR_DETAIL_PROTECT_THRESHOLD}"
  --edit-color-detail-protect-softness "${EDIT_COLOR_DETAIL_PROTECT_SOFTNESS}"
  --edit-color-target-chroma-scale "${EDIT_COLOR_TARGET_CHROMA_SCALE}"
  --edit-color-smooth-kernel "${EDIT_COLOR_SMOOTH_KERNEL}"
  --edit-color-luma-preserve-scale "${EDIT_COLOR_LUMA_PRESERVE_SCALE}"
  --edit-color-luma-gradient-preserve-scale "${EDIT_COLOR_LUMA_GRADIENT_PRESERVE_SCALE}"
  --edit-ref-guidance-scale "${EDIT_REF_GUIDANCE_SCALE}"
  --edit-ref-image "${EDIT_REF_IMAGE}"
  --edit-ref-mask "${EDIT_REF_MASK}"
  --edit-ref-structure-image "${EDIT_REF_STRUCTURE_IMAGE}"
  --edit-ref-chroma-mode "${EDIT_REF_CHROMA_MODE}"
  --edit-ref-chroma-magnitude-scale "${EDIT_REF_CHROMA_MAGNITUDE_SCALE}"
  --edit-ref-luma-preserve-scale "${EDIT_REF_LUMA_PRESERVE_SCALE}"
  --edit-ref-gradient-preserve-scale "${EDIT_REF_GRADIENT_PRESERVE_SCALE}"
  --edit-ref-darkness-guard-scale "${EDIT_REF_DARKNESS_GUARD_SCALE}"
  --edit-ref-darkness-guard-margin "${EDIT_REF_DARKNESS_GUARD_MARGIN}"
  --edit-ref-smooth-kernel "${EDIT_REF_SMOOTH_KERNEL}"
  --edit-ref-lowfreq-suppress-kernel "${EDIT_REF_LOWFREQ_SUPPRESS_KERNEL}"
  --edit-ref-lowfreq-suppress-scale "${EDIT_REF_LOWFREQ_SUPPRESS_SCALE}"
  --edit-ref-schedule-start "${EDIT_REF_SCHEDULE_START}"
  --edit-ref-schedule-stop "${EDIT_REF_SCHEDULE_STOP}"
  --edit-ref-schedule-power "${EDIT_REF_SCHEDULE_POWER}"
  --edit-ref-max-struct-rms-ratio "${EDIT_REF_MAX_STRUCT_RMS_RATIO}"
  --edit-ref-project-struct-conflict "${EDIT_REF_PROJECT_STRUCT_CONFLICT}"
  --rec-guidance-scale "${REC_GUIDANCE_SCALE}"
  --trajectory-preserve-scale "${TRAJECTORY_PRESERVE_SCALE}"
  --trajectory-subject-preserve-scale 0
  --edit-guidance-scale 0
  --edit-region-guidance-scale 0
  --edit-target-guidance-scale 0
  --edit-source-guidance-scale 0
  --edit-clip-guidance-scale 0
  --edit-dds-guidance-scale 0
  --edit-app-guidance-scale 0
  --rec-stop-timestep 0.08
  --beta-max 1.0
  --velocity-conversion-mode linear_path
  --linear-path-t-min 0.05
  --photo-prompt-mode "${PHOTO_PROMPT_MODE}"
  --log-every 7
)

if [[ -n "${EDIT_COLOR_SOURCE}" ]]; then
  cmd+=(--edit-color-source "${EDIT_COLOR_SOURCE}")
fi

if [[ -n "${EDIT_COLOR_TARGET}" ]]; then
  cmd+=(--edit-color-target "${EDIT_COLOR_TARGET}")
fi

if [[ -n "${FINAL_EDIT_MASK}" ]]; then
  cmd+=(--final-edit-mask "${FINAL_EDIT_MASK}" --final-edit-mask-mode "${FINAL_EDIT_MASK_MODE}")
fi

if [[ -n "${SUPPORT_MASK}" ]]; then
  cmd+=(--support-mask "${SUPPORT_MASK}")
fi

printf '%q ' "CUDA_VISIBLE_DEVICES=${DEVICE}" "${cmd[@]}" > "${OUT_DIR}/command.txt"
printf '\n' >> "${OUT_DIR}/command.txt"
CUDA_VISIBLE_DEVICES="${DEVICE}" "${cmd[@]}"
