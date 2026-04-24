#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
FORCE="${FORCE:-0}"

run_case() {
  local name="$1"
  local edit_hedit_scale="$2"
  local edit_anchor_scale="$3"

  local out_dir="${ROOT}/outputs/sunglasses_${name}"
  mkdir -p "${out_dir}"
  if [[ "${FORCE}" != "1" && -f "${out_dir}/result.png" && -f "${out_dir}/stats.json" ]]; then
    echo "[skip] sunglasses_${name} already exists"
    return
  fi

  local cmd=(
    "${PYTHON}" "${ROOT}/run_edit_sd3.py"
    --image "${ROOT}/h_edit_compare/panda.jpg"
    --source-prompt "A panda is walking in a forest."
    --prompt "A panda wearing sunglasses is walking in a forest."
    --output "${out_dir}/result.png"
    --stats-output "${out_dir}/stats.json"
    --metadata-output "${out_dir}/metadata.json"
    --n-max 24
    --src-guidance-scale 1.0
    --tar-guidance-scale 3.5
    --edit-hedit-guidance-scale "${edit_hedit_scale}"
    --rec-guidance-scale 0
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

run_case direct_src1_tar35 1.0 0
run_case anchor_src1_tar35 0 1.0
