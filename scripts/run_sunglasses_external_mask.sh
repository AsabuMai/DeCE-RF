#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
EXTERNAL_MASK="${EXTERNAL_MASK:-${ROOT}/outputs/proposal_composite_fireflow_v2/fireflow_face_v2_mask.png}"
TRAJECTORY_PRESERVE_SCALE="${TRAJECTORY_PRESERVE_SCALE:-0.15}"
EXTERNAL_MASK_DILATE_KERNEL="${EXTERNAL_MASK_DILATE_KERNEL:-5}"
EXTERNAL_MASK_SMOOTH_KERNEL="${EXTERNAL_MASK_SMOOTH_KERNEL:-3}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/sunglasses_external_mask}"
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/masks"

if [[ ! -f "${EXTERNAL_MASK}" ]]; then
  echo "External mask not found: ${EXTERNAL_MASK}" >&2
  echo "Set EXTERNAL_MASK=/path/to/mask.png or generate the diagnostic mask first." >&2
  exit 1
fi

CMD=(
  "${PYTHON}" "${ROOT}/run_edit_sd3.py"
  --image "${ROOT}/h_edit_compare/panda.jpg"
  --source-prompt "A panda is walking in a forest."
  --prompt "A panda wearing sunglasses over its eyes is walking in a forest."
  --output "${OUT_DIR}/result.png"
  --stats-output "${OUT_DIR}/stats.json"
  --metadata-output "${OUT_DIR}/metadata.json"
  --mask-output-dir "${OUT_DIR}/masks"
  --external-edit-mask "${EXTERNAL_MASK}"
  --external-edit-mask-mode replace
  --edit-mask-dilate-kernel "${EXTERNAL_MASK_DILATE_KERNEL}"
  --edit-mask-smooth-kernel "${EXTERNAL_MASK_SMOOTH_KERNEL}"
  --src-guidance-scale 1.0
  --tar-guidance-scale 10.5
  --attention-mask-mode target_changed
  --attention-mask-target-words sunglasses,eyes
  --attention-mask-subject-threshold 0.48
  --attention-mask-core-threshold 0.72
  --edit-hedit-guidance-scale 1.0
  --rec-guidance-scale 0
  --trajectory-preserve-scale "${TRAJECTORY_PRESERVE_SCALE}"
  --beta-max 1.0
  --velocity-conversion-mode linear_path
  --linear-path-t-min 0.05
  --rec-stop-timestep 0.08
  --photo-prompt-mode both
  --log-every 7
)

printf '%q ' "CUDA_VISIBLE_DEVICES=${DEVICE}" "${CMD[@]}" > "${OUT_DIR}/command.txt"
printf '\n' >> "${OUT_DIR}/command.txt"
CUDA_VISIBLE_DEVICES="${DEVICE}" "${CMD[@]}"
