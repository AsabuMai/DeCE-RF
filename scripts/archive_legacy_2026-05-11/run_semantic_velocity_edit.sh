#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"

IMAGE="${IMAGE:-${ROOT}/h_edit_compare/panda.jpg}"
SOURCE_PROMPT="${SOURCE_PROMPT:-A panda is walking in a forest.}"
TARGET_PROMPT="${TARGET_PROMPT:-A panda wearing small black sunglasses is walking in a forest.}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/semantic_velocity_edit}"

SUPPORT_MASK="${SUPPORT_MASK:-${SEMANTIC_BASE_MASK:-}}"
GROUNDING_MODEL="${GROUNDING_MODEL:-IDEA-Research/grounding-dino-base}"
SAM_MODEL="${SAM_MODEL:-facebook/sam-vit-base}"
SEMANTIC_PHRASE="${SEMANTIC_PHRASE:-}"
SUPPORT_RELATION="${SUPPORT_RELATION:-auto}"
SUPPORT_EXPAND_X="${SUPPORT_EXPAND_X:-0.0}"
SUPPORT_EXPAND_Y="${SUPPORT_EXPAND_Y:-0.0}"
SUPPORT_BAND_RATIO="${SUPPORT_BAND_RATIO:-0.55}"
SUPPORT_OVERLAP_RATIO="${SUPPORT_OVERLAP_RATIO:-0.20}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.2}"
EYES_ANCHOR_MAX_AREA_RATIO="${EYES_ANCHOR_MAX_AREA_RATIO:-0.16}"
EYES_ANCHOR_MAX_BOX_AREA_RATIO="${EYES_ANCHOR_MAX_BOX_AREA_RATIO:-0.34}"

mkdir -p "${OUT_DIR}/masks"

if [[ -z "${SUPPORT_MASK}" ]]; then
  mask_cmd=(
    "${PYTHON}" "${ROOT}/scripts/make_semantic_mask.py"
    --image "${IMAGE}"
    --source-prompt "${SOURCE_PROMPT}"
    --prompt "${TARGET_PROMPT}"
    --output "${OUT_DIR}/masks/semantic_base_generated.png"
    --metadata-output "${OUT_DIR}/masks/semantic_base_generated.json"
    --anchor-output "${OUT_DIR}/masks/semantic_anchor_generated.png"
    --device "cuda:${DEVICE}"
    --grounding-model "${GROUNDING_MODEL}"
    --sam-model "${SAM_MODEL}"
    --support-relation "${SUPPORT_RELATION}"
    --support-expand-x "${SUPPORT_EXPAND_X}"
    --support-expand-y "${SUPPORT_EXPAND_Y}"
    --support-band-ratio "${SUPPORT_BAND_RATIO}"
    --support-overlap-ratio "${SUPPORT_OVERLAP_RATIO}"
    --support-threshold "${SUPPORT_THRESHOLD}"
    --eyes-anchor-max-area-ratio "${EYES_ANCHOR_MAX_AREA_RATIO}"
    --eyes-anchor-max-box-area-ratio "${EYES_ANCHOR_MAX_BOX_AREA_RATIO}"
  )
  if [[ -n "${SEMANTIC_PHRASE}" ]]; then
    mask_cmd+=(--phrase "${SEMANTIC_PHRASE}")
  fi
  "${mask_cmd[@]}"
  SUPPORT_MASK="${OUT_DIR}/masks/semantic_base_generated.png"
fi

OBJECT_MASK_PROVIDER=semantic_velocity \
SUPPORT_MASK="${SUPPORT_MASK}" \
DEVICE="${DEVICE}" \
IMAGE="${IMAGE}" \
SOURCE_PROMPT="${SOURCE_PROMPT}" \
TARGET_PROMPT="${TARGET_PROMPT}" \
OUT_DIR="${OUT_DIR}" \
"${ROOT}/scripts/run_ode_decoupled_edit.sh"
