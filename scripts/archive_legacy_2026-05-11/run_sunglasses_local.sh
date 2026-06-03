#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
IMAGE="${IMAGE:-${ROOT}/h_edit_compare/panda.jpg}"
SOURCE_PROMPT="${SOURCE_PROMPT:-A panda is walking in a forest.}"
TARGET_PROMPT="${TARGET_PROMPT:-A panda wearing small black sunglasses is walking in a forest.}"
ATTENTION_MASK_MODE="${ATTENTION_MASK_MODE:-target_changed}"
ATTENTION_MASK_TARGET_WORDS="${ATTENTION_MASK_TARGET_WORDS:-}"
ATTENTION_MASK_MAX_AREA_RATIO="${ATTENTION_MASK_MAX_AREA_RATIO:-0.25}"
ATTENTION_MASK_FALLBACK_THRESHOLD="${ATTENTION_MASK_FALLBACK_THRESHOLD:-0.72}"
OBJECT_MASK_PROVIDER="${OBJECT_MASK_PROVIDER:-attention_velocity}"
PROPOSAL_EDIT_IMAGE="${PROPOSAL_EDIT_IMAGE:-}"
PROPOSAL_MASK_THRESHOLD="${PROPOSAL_MASK_THRESHOLD:-0.22}"
PROPOSAL_MASK_KEEP_COMPONENTS="${PROPOSAL_MASK_KEEP_COMPONENTS:-2}"
PROPOSAL_MASK_MIN_AREA="${PROPOSAL_MASK_MIN_AREA:-24}"
PROPOSAL_MASK_DILATE="${PROPOSAL_MASK_DILATE:-9}"
PROPOSAL_MASK_ERODE="${PROPOSAL_MASK_ERODE:-0}"
PROPOSAL_MASK_BLUR="${PROPOSAL_MASK_BLUR:-17}"
PROPOSAL_MASK_DARK_BIAS="${PROPOSAL_MASK_DARK_BIAS:-1.0}"
MASK_LAYERING_MODE="${MASK_LAYERING_MODE:-object_contact}"
MASK_OBJECT_THRESHOLD="${MASK_OBJECT_THRESHOLD:-0.45}"
MASK_CONTACT_DILATE_KERNEL="${MASK_CONTACT_DILATE_KERNEL:-7}"
MASK_CONTACT_SCALE="${MASK_CONTACT_SCALE:-0.25}"
MASK_CONTACT_EDGE_THRESHOLD="${MASK_CONTACT_EDGE_THRESHOLD:-0.55}"
MASK_CONTACT_EDGE_PROTECT_SCALE="${MASK_CONTACT_EDGE_PROTECT_SCALE:-0.75}"
PHOTO_PROMPT_MODE="${PHOTO_PROMPT_MODE:-both}"
TRAJECTORY_PRESERVE_SCALE="${TRAJECTORY_PRESERVE_SCALE:-0.15}"
TRAJECTORY_SUBJECT_PRESERVE_SCALE="${TRAJECTORY_SUBJECT_PRESERVE_SCALE:-0.0}"
EDIT_INITIAL_NOISE_SCALE="${EDIT_INITIAL_NOISE_SCALE:-0.0}"
SRC_GUIDANCE_SCALE="${SRC_GUIDANCE_SCALE:-1.0}"
TAR_GUIDANCE_SCALE="${TAR_GUIDANCE_SCALE:-10.5}"
EDIT_HEDIT_GUIDANCE_SCALE="${EDIT_HEDIT_GUIDANCE_SCALE:-0.75}"
EDIT_CORE_SCALE="${EDIT_CORE_SCALE:-1.35}"
EDIT_SUBJECT_SCALE="${EDIT_SUBJECT_SCALE:-0.35}"
SOURCE_INJECT_Q_SCALE="${SOURCE_INJECT_Q_SCALE:-0.0}"
SOURCE_INJECT_K_SCALE="${SOURCE_INJECT_K_SCALE:-0.25}"
SOURCE_INJECT_V_SCALE="${SOURCE_INJECT_V_SCALE:-0.0}"
SOURCE_INJECT_LAYER_FROM="${SOURCE_INJECT_LAYER_FROM:--1}"
SOURCE_INJECT_LAYER_TO="${SOURCE_INJECT_LAYER_TO:--1}"
SOURCE_INJECT_STEPS="${SOURCE_INJECT_STEPS:-8}"
SOURCE_INJECT_MASK_MODE="${SOURCE_INJECT_MASK_MODE:-core}"
SOURCE_INJECT_MASK_BOX="${SOURCE_INJECT_MASK_BOX:-}"
AUTO_LOCAL_BOXES="${AUTO_LOCAL_BOXES:-0}"
AUTO_BOX_THRESHOLD="${AUTO_BOX_THRESHOLD:-0.35}"
AUTO_EDIT_PAD_X="${AUTO_EDIT_PAD_X:-0.08}"
AUTO_EDIT_PAD_Y="${AUTO_EDIT_PAD_Y:-0.04}"
AUTO_EDIT_MIN_WIDTH="${AUTO_EDIT_MIN_WIDTH:-0.28}"
AUTO_EDIT_MIN_HEIGHT="${AUTO_EDIT_MIN_HEIGHT:-0.10}"
AUTO_SOURCE_PAD_X="${AUTO_SOURCE_PAD_X:-0.14}"
AUTO_SOURCE_PAD_Y="${AUTO_SOURCE_PAD_Y:-0.08}"
AUTO_PRESERVE_PAD_X="${AUTO_PRESERVE_PAD_X:-0.04}"
AUTO_PRESERVE_START_OFFSET="${AUTO_PRESERVE_START_OFFSET:-0.06}"
AUTO_PRESERVE_HEIGHT="${AUTO_PRESERVE_HEIGHT:-0.24}"
AUTO_STRUCTURE_BOXES="${AUTO_STRUCTURE_BOXES:-0}"
AUTO_STRUCTURE_MODE="${AUTO_STRUCTURE_MODE:-dark_eyes}"
AUTO_STRUCTURE_EXTERNAL_MASK="${AUTO_STRUCTURE_EXTERNAL_MASK:-0}"
STRUCTURE_GLASSES_ANGLE_MODE="${STRUCTURE_GLASSES_ANGLE_MODE:-zero}"
STRUCTURE_GLASSES_MAX_ANGLE="${STRUCTURE_GLASSES_MAX_ANGLE:-8.0}"
if [[ "${AUTO_STRUCTURE_EXTERNAL_MASK}" == "1" ]]; then
  EDIT_MASK_DILATE_KERNEL="${EDIT_MASK_DILATE_KERNEL:-1}"
  EDIT_MASK_SMOOTH_KERNEL="${EDIT_MASK_SMOOTH_KERNEL:-3}"
else
  EDIT_MASK_DILATE_KERNEL="${EDIT_MASK_DILATE_KERNEL:-3}"
  EDIT_MASK_SMOOTH_KERNEL="${EDIT_MASK_SMOOTH_KERNEL:-3}"
fi
if [[ "${AUTO_STRUCTURE_BOXES}" == "1" ]]; then
  MASK_SHIFT_Y="${MASK_SHIFT_Y:-0.0}"
else
  MASK_SHIFT_Y="${MASK_SHIFT_Y:-0.0}"
fi
MASK_COMPONENT_Y_MAX="${MASK_COMPONENT_Y_MAX:-}"
MASK_KEEP_COMPONENTS="${MASK_KEEP_COMPONENTS:-}"
EDIT_MASK_BOX="${EDIT_MASK_BOX:-}"
EDIT_MASK_BOX_MODE="${EDIT_MASK_BOX_MODE:-union}"
EDIT_MASK_EXCLUDE_BOX="${EDIT_MASK_EXCLUDE_BOX:-}"
USE_CORE_AS_EDIT_MASK="${USE_CORE_AS_EDIT_MASK:-0}"
MASK_BLEND="${MASK_BLEND:-0}"
MASK_BLEND_MODE="${MASK_BLEND_MODE:-subject}"
FINAL_PRESERVE_BOX="${FINAL_PRESERVE_BOX:-}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/sunglasses_local}"
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/masks"

CMD=(
  "${PYTHON}" "${ROOT}/run_edit_sd3.py"
  --image "${IMAGE}"
  --source-prompt "${SOURCE_PROMPT}"
  --prompt "${TARGET_PROMPT}"
  --output "${OUT_DIR}/result.png"
  --stats-output "${OUT_DIR}/stats.json"
  --metadata-output "${OUT_DIR}/metadata.json"
  --mask-output-dir "${OUT_DIR}/masks"
  --src-guidance-scale "${SRC_GUIDANCE_SCALE}"
  --tar-guidance-scale "${TAR_GUIDANCE_SCALE}"
  --attention-mask-mode "${ATTENTION_MASK_MODE}"
  --attention-mask-target-words "${ATTENTION_MASK_TARGET_WORDS}"
  --attention-mask-subject-threshold 0.48
  --attention-mask-core-threshold 0.72
  --attention-mask-max-area-ratio "${ATTENTION_MASK_MAX_AREA_RATIO}"
  --attention-mask-fallback-threshold "${ATTENTION_MASK_FALLBACK_THRESHOLD}"
  --object-mask-provider "${OBJECT_MASK_PROVIDER}"
  --proposal-mask-threshold "${PROPOSAL_MASK_THRESHOLD}"
  --proposal-mask-keep-components "${PROPOSAL_MASK_KEEP_COMPONENTS}"
  --proposal-mask-min-area "${PROPOSAL_MASK_MIN_AREA}"
  --proposal-mask-dilate "${PROPOSAL_MASK_DILATE}"
  --proposal-mask-erode "${PROPOSAL_MASK_ERODE}"
  --proposal-mask-blur "${PROPOSAL_MASK_BLUR}"
  --proposal-mask-dark-bias "${PROPOSAL_MASK_DARK_BIAS}"
  --mask-layering-mode "${MASK_LAYERING_MODE}"
  --mask-object-threshold "${MASK_OBJECT_THRESHOLD}"
  --mask-contact-dilate-kernel "${MASK_CONTACT_DILATE_KERNEL}"
  --mask-contact-scale "${MASK_CONTACT_SCALE}"
  --mask-contact-edge-threshold "${MASK_CONTACT_EDGE_THRESHOLD}"
  --mask-contact-edge-protect-scale "${MASK_CONTACT_EDGE_PROTECT_SCALE}"
  --edit-mask-dilate-kernel "${EDIT_MASK_DILATE_KERNEL}"
  --edit-mask-smooth-kernel "${EDIT_MASK_SMOOTH_KERNEL}"
  --edit-mask-shift-y "${MASK_SHIFT_Y}"
  --edit-mask-box-mode "${EDIT_MASK_BOX_MODE}"
  --edit-hedit-guidance-scale "${EDIT_HEDIT_GUIDANCE_SCALE}"
  --edit-core-scale "${EDIT_CORE_SCALE}"
  --edit-subject-scale "${EDIT_SUBJECT_SCALE}"
  --source-inject-q-scale "${SOURCE_INJECT_Q_SCALE}"
  --source-inject-k-scale "${SOURCE_INJECT_K_SCALE}"
  --source-inject-v-scale "${SOURCE_INJECT_V_SCALE}"
  --source-inject-layer-from "${SOURCE_INJECT_LAYER_FROM}"
  --source-inject-layer-to "${SOURCE_INJECT_LAYER_TO}"
  --source-inject-steps "${SOURCE_INJECT_STEPS}"
  --source-inject-mask-mode "${SOURCE_INJECT_MASK_MODE}"
  --rec-guidance-scale 0
  --trajectory-preserve-scale "${TRAJECTORY_PRESERVE_SCALE}"
  --trajectory-subject-preserve-scale "${TRAJECTORY_SUBJECT_PRESERVE_SCALE}"
  --edit-initial-noise-scale "${EDIT_INITIAL_NOISE_SCALE}"
  --edit-initial-noise-region core
  --beta-max 1.0
  --velocity-conversion-mode linear_path
  --linear-path-t-min 0.05
  --rec-stop-timestep 0.08
  --photo-prompt-mode "${PHOTO_PROMPT_MODE}"
  --log-every 7
)

if [[ "${AUTO_LOCAL_BOXES}" == "1" ]]; then
  CMD+=(
    --auto-local-boxes
    --auto-box-threshold "${AUTO_BOX_THRESHOLD}"
    --auto-edit-pad-x "${AUTO_EDIT_PAD_X}"
    --auto-edit-pad-y "${AUTO_EDIT_PAD_Y}"
    --auto-edit-min-width "${AUTO_EDIT_MIN_WIDTH}"
    --auto-edit-min-height "${AUTO_EDIT_MIN_HEIGHT}"
    --auto-source-pad-x "${AUTO_SOURCE_PAD_X}"
    --auto-source-pad-y "${AUTO_SOURCE_PAD_Y}"
    --auto-preserve-pad-x "${AUTO_PRESERVE_PAD_X}"
    --auto-preserve-start-offset "${AUTO_PRESERVE_START_OFFSET}"
    --auto-preserve-height "${AUTO_PRESERVE_HEIGHT}"
  )
fi

if [[ -n "${PROPOSAL_EDIT_IMAGE}" ]]; then
  CMD+=(
    --proposal-edit-image "${PROPOSAL_EDIT_IMAGE}"
  )
fi

if [[ "${AUTO_STRUCTURE_BOXES}" == "1" ]]; then
  CMD+=(
    --auto-structure-boxes
    --auto-structure-mode "${AUTO_STRUCTURE_MODE}"
    --structure-glasses-angle-mode "${STRUCTURE_GLASSES_ANGLE_MODE}"
    --structure-glasses-max-angle "${STRUCTURE_GLASSES_MAX_ANGLE}"
  )
  if [[ "${AUTO_STRUCTURE_EXTERNAL_MASK}" == "1" ]]; then
    CMD+=(--auto-structure-external-mask)
  fi
fi

if [[ -n "${MASK_COMPONENT_Y_MAX}" ]]; then
  CMD+=(
    --edit-mask-component-y-max "${MASK_COMPONENT_Y_MAX}"
    --edit-mask-component-threshold 0.5
  )
fi

if [[ -n "${MASK_KEEP_COMPONENTS}" ]]; then
  CMD+=(
    --edit-mask-keep-components "${MASK_KEEP_COMPONENTS}"
    --edit-mask-component-threshold 0.5
  )
fi

if [[ -n "${EDIT_MASK_BOX}" ]]; then
  CMD+=(
    --edit-mask-box "${EDIT_MASK_BOX}"
    --edit-mask-box-mode "${EDIT_MASK_BOX_MODE}"
  )
fi

if [[ -n "${EDIT_MASK_EXCLUDE_BOX}" ]]; then
  CMD+=(
    --edit-mask-exclude-box "${EDIT_MASK_EXCLUDE_BOX}"
  )
fi

if [[ -n "${SOURCE_INJECT_MASK_BOX}" ]]; then
  CMD+=(
    --source-inject-mask-box "${SOURCE_INJECT_MASK_BOX}"
  )
fi

if [[ "${USE_CORE_AS_EDIT_MASK}" == "1" ]]; then
  CMD+=(--edit-mask-use-core-as-subject)
fi

if [[ "${MASK_BLEND}" == "1" ]]; then
  CMD+=(
    --mask-blend
    --mask-blend-mode "${MASK_BLEND_MODE}"
  )
fi

if [[ -n "${FINAL_PRESERVE_BOX}" ]]; then
  CMD+=(
    --final-preserve-box "${FINAL_PRESERVE_BOX}"
  )
fi

printf '%q ' "CUDA_VISIBLE_DEVICES=${DEVICE}" "${CMD[@]}" > "${OUT_DIR}/command.txt"
printf '\n' >> "${OUT_DIR}/command.txt"
CUDA_VISIBLE_DEVICES="${DEVICE}" "${CMD[@]}"
