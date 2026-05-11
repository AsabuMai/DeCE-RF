#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
IMAGE="${IMAGE:-${ROOT}/h_edit_compare/panda.jpg}"
SOURCE_PROMPT="${SOURCE_PROMPT:-A panda is walking in a forest.}"
TARGET_PROMPT="${TARGET_PROMPT:-A panda wearing small black sunglasses is walking in a forest.}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/decoupling_core}"
FIXED_EDIT_MASK="${FIXED_EDIT_MASK:-${ROOT}/outputs/sunglasses_effect_good_nomasklayer_20260504/masks/structure_glasses_edit_mask.png}"

SRC_GUIDANCE_SCALE="${SRC_GUIDANCE_SCALE:-1.0}"
TAR_GUIDANCE_SCALE="${TAR_GUIDANCE_SCALE:-10.5}"
EDIT_HEDIT_GUIDANCE_SCALE="${EDIT_HEDIT_GUIDANCE_SCALE:-0.75}"
REC_GUIDANCE_SCALE="${REC_GUIDANCE_SCALE:-0.25}"
TRAJECTORY_PRESERVE_SCALE="${TRAJECTORY_PRESERVE_SCALE:-0.15}"
STRUCT_GUIDANCE_SCALE="${STRUCT_GUIDANCE_SCALE:-0.0}"
EDIT_CORE_SCALE="${EDIT_CORE_SCALE:-1.35}"
EDIT_SUBJECT_SCALE="${EDIT_SUBJECT_SCALE:-0.35}"

if [[ ! -f "${FIXED_EDIT_MASK}" ]]; then
  echo "Fixed edit mask not found: ${FIXED_EDIT_MASK}" >&2
  echo "Set FIXED_EDIT_MASK=/path/to/mask.png or run scripts/run_sunglasses_good.sh first." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

run_case() {
  local name="$1"
  local edit_scale="$2"
  local rec_scale="$3"
  local traj_scale="$4"

  local out_dir="${OUT_ROOT}/${name}"
  mkdir -p "${out_dir}/masks"

  local cmd=(
    "${PYTHON}" "${ROOT}/run_edit_sd3.py"
    --image "${IMAGE}"
    --source-prompt "${SOURCE_PROMPT}"
    --prompt "${TARGET_PROMPT}"
    --output "${out_dir}/result.png"
    --stats-output "${out_dir}/stats.json"
    --metadata-output "${out_dir}/metadata.json"
    --mask-output-dir "${out_dir}/masks"
    --external-edit-mask "${FIXED_EDIT_MASK}"
    --external-edit-mask-mode replace
    --mask-layering-mode none
    --edit-mask-dilate-kernel 1
    --edit-mask-smooth-kernel 3
    --src-guidance-scale "${SRC_GUIDANCE_SCALE}"
    --tar-guidance-scale "${TAR_GUIDANCE_SCALE}"
    --edit-hedit-guidance-scale "${edit_scale}"
    --edit-core-scale "${EDIT_CORE_SCALE}"
    --edit-subject-scale "${EDIT_SUBJECT_SCALE}"
    --rec-guidance-scale "${rec_scale}"
    --struct-guidance-scale "${STRUCT_GUIDANCE_SCALE}"
    --trajectory-preserve-scale "${traj_scale}"
    --trajectory-subject-preserve-scale 0
    --edit-guidance-scale 0
    --edit-region-guidance-scale 0
    --edit-target-guidance-scale 0
    --edit-source-guidance-scale 0
    --edit-clip-guidance-scale 0
    --edit-text-guidance-scale 0
    --edit-dds-guidance-scale 0
    --edit-app-guidance-scale 0
    --rec-stop-timestep 0.08
    --beta-max 1.0
    --velocity-conversion-mode linear_path
    --linear-path-t-min 0.05
    --photo-prompt-mode both
    --log-every 7
  )

  printf '%q ' "CUDA_VISIBLE_DEVICES=${DEVICE}" "${cmd[@]}" > "${out_dir}/command.txt"
  printf '\n' >> "${out_dir}/command.txt"
  CUDA_VISIBLE_DEVICES="${DEVICE}" "${cmd[@]}"
}

run_case base_only 0 0 0
run_case edit_only "${EDIT_HEDIT_GUIDANCE_SCALE}" 0 0
run_case rec_only 0 "${REC_GUIDANCE_SCALE}" "${TRAJECTORY_PRESERVE_SCALE}"
run_case decoupled "${EDIT_HEDIT_GUIDANCE_SCALE}" "${REC_GUIDANCE_SCALE}" "${TRAJECTORY_PRESERVE_SCALE}"

"${PYTHON}" "${ROOT}/scripts/make_comparison_grid.py" \
  --output "${OUT_ROOT}/comparison_grid.png" \
  source="${IMAGE}" \
  base="${OUT_ROOT}/base_only/result.png" \
  edit_only="${OUT_ROOT}/edit_only/result.png" \
  rec_only="${OUT_ROOT}/rec_only/result.png" \
  decoupled="${OUT_ROOT}/decoupled/result.png"

cat > "${OUT_ROOT}/metadata.json" <<JSON
{
  "description": "Fixed-support minimal decoupling experiment. All cases use the same source, target prompt, RF parameters, and external edit mask; only u_rec/u_edit switches change.",
  "image": "${IMAGE}",
  "source_prompt": "${SOURCE_PROMPT}",
  "target_prompt": "${TARGET_PROMPT}",
  "fixed_edit_mask": "${FIXED_EDIT_MASK}",
  "cases": {
    "base_only": {"edit_hedit_guidance_scale": 0, "rec_guidance_scale": 0, "trajectory_preserve_scale": 0},
    "edit_only": {"edit_hedit_guidance_scale": ${EDIT_HEDIT_GUIDANCE_SCALE}, "rec_guidance_scale": 0, "trajectory_preserve_scale": 0},
    "rec_only": {"edit_hedit_guidance_scale": 0, "rec_guidance_scale": ${REC_GUIDANCE_SCALE}, "trajectory_preserve_scale": ${TRAJECTORY_PRESERVE_SCALE}},
    "decoupled": {"edit_hedit_guidance_scale": ${EDIT_HEDIT_GUIDANCE_SCALE}, "rec_guidance_scale": ${REC_GUIDANCE_SCALE}, "trajectory_preserve_scale": ${TRAJECTORY_PRESERVE_SCALE}}
  }
}
JSON
