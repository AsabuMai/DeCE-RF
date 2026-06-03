#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
TRAJECTORY_PRESERVE_SCALE="${TRAJECTORY_PRESERVE_SCALE:-0.15}"
TRAJECTORY_SUBJECT_PRESERVE_SCALE="${TRAJECTORY_SUBJECT_PRESERVE_SCALE:-0.0}"
EDIT_INITIAL_NOISE_SCALE="${EDIT_INITIAL_NOISE_SCALE:-0.0}"
MASK_SHIFT_Y="${MASK_SHIFT_Y:-0.10}"
MASK_COMPONENT_Y_MAX="${MASK_COMPONENT_Y_MAX:-0.45}"
OUT_DIR="${ROOT}/outputs/sunglasses_trajectory_preserve"
mkdir -p "${OUT_DIR}"

CMD=(
  "${PYTHON}" "${ROOT}/run_edit_sd3.py"
  --image "${ROOT}/h_edit_compare/panda.jpg"
  --source-prompt "A panda is walking in a forest."
  --prompt "A panda wearing sunglasses over its eyes is walking in a forest."
  --output "${OUT_DIR}/result.png"
  --stats-output "${OUT_DIR}/stats.json"
  --metadata-output "${OUT_DIR}/metadata.json"
  --mask-output-dir "${OUT_DIR}/masks"
  --src-guidance-scale 1.0
  --tar-guidance-scale 10.5
  --attention-mask-mode target_changed
  --attention-mask-target-words sunglasses,eyes
  --attention-mask-subject-threshold 0.48
  --attention-mask-core-threshold 0.72
  --edit-mask-dilate-kernel 5
  --edit-mask-smooth-kernel 5
  --edit-mask-shift-y "${MASK_SHIFT_Y}"
  --edit-hedit-guidance-scale 1.0
  --rec-guidance-scale 0
  --trajectory-preserve-scale "${TRAJECTORY_PRESERVE_SCALE}"
  --trajectory-subject-preserve-scale "${TRAJECTORY_SUBJECT_PRESERVE_SCALE}"
  --edit-initial-noise-scale "${EDIT_INITIAL_NOISE_SCALE}"
  --edit-initial-noise-region core
  --beta-max 1.0
  --velocity-conversion-mode linear_path
  --linear-path-t-min 0.05
  --rec-stop-timestep 0.08
  --photo-prompt-mode both
  --log-every 7
)

if [[ -n "${MASK_COMPONENT_Y_MAX}" ]]; then
  CMD+=(
    --edit-mask-component-y-max "${MASK_COMPONENT_Y_MAX}"
    --edit-mask-component-threshold 0.5
  )
fi

printf '%q ' "CUDA_VISIBLE_DEVICES=${DEVICE}" "${CMD[@]}" > "${OUT_DIR}/command.txt"
printf '\n' >> "${OUT_DIR}/command.txt"
CUDA_VISIBLE_DEVICES="${DEVICE}" "${CMD[@]}"
