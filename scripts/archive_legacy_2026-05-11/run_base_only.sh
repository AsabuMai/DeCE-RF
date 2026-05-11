#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
OUT_DIR="${ROOT}/outputs/base_only"
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/masks"

CMD=(
  "${PYTHON}" "${ROOT}/run_edit_sd3.py"
  --image "${ROOT}/h_edit_compare/panda.jpg"
  --source-prompt "A panda is walking in a forest."
  --prompt "A tiger is walking in a forest."
  --output "${OUT_DIR}/result.png"
  --stats-output "${OUT_DIR}/stats.json"
  --metadata-output "${OUT_DIR}/metadata.json"
  --mask-output-dir "${OUT_DIR}/masks"
  --src-guidance-scale 1.0
  --tar-guidance-scale 10.5
  --edit-hedit-guidance-scale 0
  --edit-guidance-scale 0
  --edit-region-guidance-scale 0
  --edit-target-guidance-scale 0
  --edit-source-guidance-scale 0
  --edit-clip-guidance-scale 0
  --edit-text-guidance-scale 0
  --edit-dds-guidance-scale 0
  --edit-app-guidance-scale 0
  --rec-guidance-scale 0
  --beta-max 1.0
  --velocity-conversion-mode linear_path
  --linear-path-t-min 0.05
  --rec-stop-timestep 0.08
  --photo-prompt-mode both
  --log-every 1
)

printf '%q ' "CUDA_VISIBLE_DEVICES=${DEVICE}" "${CMD[@]}" > "${OUT_DIR}/command.txt"
printf '\n' >> "${OUT_DIR}/command.txt"
CUDA_VISIBLE_DEVICES="${DEVICE}" "${CMD[@]}"
