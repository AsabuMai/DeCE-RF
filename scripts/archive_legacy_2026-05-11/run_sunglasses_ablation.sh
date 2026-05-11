#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
RUN_BASE="${RUN_BASE:-0}"

run_case() {
  local name="$1"
  local edit_hedit_scale="$2"
  local rec_scale="$3"
  local struct_scale="$4"
  local edit_anchor_scale="$5"

  local out_dir="${ROOT}/outputs/sunglasses_${name}"
  mkdir -p "${out_dir}"
  mkdir -p "${out_dir}/masks"

  local cmd=(
    "${PYTHON}" "${ROOT}/run_edit_sd3.py"
    --image "${ROOT}/h_edit_compare/panda.jpg"
    --source-prompt "A panda is walking in a forest."
    --prompt "A panda wearing sunglasses is walking in a forest."
    --output "${out_dir}/result.png"
    --stats-output "${out_dir}/stats.json"
    --metadata-output "${out_dir}/metadata.json"
    --mask-output-dir "${out_dir}/masks"
    --src-guidance-scale 1.0
    --tar-guidance-scale 10.5
    --edit-hedit-guidance-scale "${edit_hedit_scale}"
    --rec-guidance-scale "${rec_scale}"
    --struct-guidance-scale "${struct_scale}"
    --edit-guidance-scale "${edit_anchor_scale}"
    --edit-region-guidance-scale 0
    --edit-target-guidance-scale 0
    --edit-source-guidance-scale 0
    --edit-clip-guidance-scale 0
    --edit-text-guidance-scale 0
    --edit-dds-guidance-scale 0
    --edit-app-guidance-scale 0
    --beta-max 1.0
    --velocity-conversion-mode linear_path
    --linear-path-t-min 0.05
    --rec-stop-timestep 0.08
    --photo-prompt-mode both
    --log-every 7
  )

  printf '%q ' "CUDA_VISIBLE_DEVICES=${DEVICE}" "${cmd[@]}" > "${out_dir}/command.txt"
  printf '\n' >> "${out_dir}/command.txt"
  CUDA_VISIBLE_DEVICES="${DEVICE}" "${cmd[@]}"
}

if [[ "${RUN_BASE}" == "1" ]]; then
  run_case base_only 0 0 0 0
fi

run_case direct_target 1.0 0 0 0
run_case decoupled_rec 1.0 0.25 0.5 0
run_case anchor_only 0 0 0 1.0
