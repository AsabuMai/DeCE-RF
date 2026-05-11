#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"

IMAGE="${IMAGE:-${ROOT}/h_edit_compare/panda.jpg}"
SOURCE_PROMPT="${SOURCE_PROMPT:-A panda is walking in a forest.}"
TARGET_PROMPT="${TARGET_PROMPT:-A panda wearing small black sunglasses is walking in a forest.}"
SEMANTIC_BASE_MASK="${SEMANTIC_BASE_MASK:-${ROOT}/outputs/semantic_mask_download_20260504/eyes_box_support_mask.png}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/semantic_velocity_ablation}"

EDIT_HEDIT_GUIDANCE_SCALE="${EDIT_HEDIT_GUIDANCE_SCALE:-0.55}"
EDIT_TEXT_GUIDANCE_SCALE="${EDIT_TEXT_GUIDANCE_SCALE:-0.08}"
REC_GUIDANCE_SCALE="${REC_GUIDANCE_SCALE:-0.25}"
TRAJECTORY_PRESERVE_SCALE="${TRAJECTORY_PRESERVE_SCALE:-0.15}"

if [[ ! -f "${SEMANTIC_BASE_MASK}" ]]; then
  echo "Semantic base mask not found: ${SEMANTIC_BASE_MASK}" >&2
  echo "Generate one with scripts/make_semantic_mask.py or set SEMANTIC_BASE_MASK=/path/to/mask.png." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

run_case() {
  local name="$1"
  local provider="$2"
  local rec_scale="$3"
  local traj_scale="$4"
  local semantic_mask="$5"

  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}"

  local env_cmd=(
    DEVICE="${DEVICE}"
    IMAGE="${IMAGE}"
    SOURCE_PROMPT="${SOURCE_PROMPT}"
    TARGET_PROMPT="${TARGET_PROMPT}"
    OUT_DIR="${out_dir}"
    OBJECT_MASK_PROVIDER="${provider}"
    MASK_LAYERING_MODE=none
    EDIT_HEDIT_GUIDANCE_SCALE="${EDIT_HEDIT_GUIDANCE_SCALE}"
    EDIT_TEXT_GUIDANCE_SCALE="${EDIT_TEXT_GUIDANCE_SCALE}"
    REC_GUIDANCE_SCALE="${rec_scale}"
    TRAJECTORY_PRESERVE_SCALE="${traj_scale}"
  )
  if [[ -n "${semantic_mask}" ]]; then
    env_cmd+=(SEMANTIC_BASE_MASK="${semantic_mask}")
  fi

  printf '%q ' "${env_cmd[@]}" "${ROOT}/scripts/run_ode_decoupled_edit.sh" > "${out_dir}/command.txt"
  printf '\n' >> "${out_dir}/command.txt"
  env "${env_cmd[@]}" "${ROOT}/scripts/run_ode_decoupled_edit.sh"
}

run_case attention_velocity attention_velocity "${REC_GUIDANCE_SCALE}" "${TRAJECTORY_PRESERVE_SCALE}" ""
run_case semantic_only semantic "${REC_GUIDANCE_SCALE}" "${TRAJECTORY_PRESERVE_SCALE}" "${SEMANTIC_BASE_MASK}"
run_case semantic_velocity_no_rec semantic_velocity 0 0 "${SEMANTIC_BASE_MASK}"
run_case semantic_velocity semantic_velocity "${REC_GUIDANCE_SCALE}" "${TRAJECTORY_PRESERVE_SCALE}" "${SEMANTIC_BASE_MASK}"

"${PYTHON}" "${ROOT}/scripts/make_comparison_grid.py" \
  --output "${OUT_ROOT}/comparison_grid.png" \
  source="${IMAGE}" \
  attention_velocity="${OUT_ROOT}/attention_velocity/result.png" \
  semantic_only="${OUT_ROOT}/semantic_only/result.png" \
  semantic_velocity_no_rec="${OUT_ROOT}/semantic_velocity_no_rec/result.png" \
  semantic_velocity="${OUT_ROOT}/semantic_velocity/result.png"

cat > "${OUT_ROOT}/metadata.json" <<JSON
{
  "description": "Semantic support ablation. Same source, prompt, edit field, and RF settings; only support provider and reconstruction switch change.",
  "image": "${IMAGE}",
  "source_prompt": "${SOURCE_PROMPT}",
  "target_prompt": "${TARGET_PROMPT}",
  "semantic_base_mask": "${SEMANTIC_BASE_MASK}",
  "edit_hedit_guidance_scale": ${EDIT_HEDIT_GUIDANCE_SCALE},
  "edit_text_guidance_scale": ${EDIT_TEXT_GUIDANCE_SCALE},
  "cases": {
    "attention_velocity": {"object_mask_provider": "attention_velocity", "rec_guidance_scale": ${REC_GUIDANCE_SCALE}, "trajectory_preserve_scale": ${TRAJECTORY_PRESERVE_SCALE}},
    "semantic_only": {"object_mask_provider": "semantic", "rec_guidance_scale": ${REC_GUIDANCE_SCALE}, "trajectory_preserve_scale": ${TRAJECTORY_PRESERVE_SCALE}},
    "semantic_velocity_no_rec": {"object_mask_provider": "semantic_velocity", "rec_guidance_scale": 0, "trajectory_preserve_scale": 0},
    "semantic_velocity": {"object_mask_provider": "semantic_velocity", "rec_guidance_scale": ${REC_GUIDANCE_SCALE}, "trajectory_preserve_scale": ${TRAJECTORY_PRESERVE_SCALE}}
  }
}
JSON
