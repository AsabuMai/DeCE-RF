#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
FORCE="${FORCE:-0}"

run_case() {
  local name="$1"
  local n_max="$2"
  local src_cfg="$3"

  local out_dir="${ROOT}/outputs/recon_${name}"
  mkdir -p "${out_dir}"

  if [[ "${FORCE}" != "1" && -f "${out_dir}/result.png" && -f "${out_dir}/stats.json" ]]; then
    echo "[skip] recon_${name} already exists"
    return
  fi

  local cmd=(
    "${PYTHON}" "${ROOT}/run_edit_sd3.py"
    --image "${ROOT}/h_edit_compare/panda.jpg"
    --source-prompt "A panda is walking in a forest."
    --prompt "A panda is walking in a forest."
    --output "${out_dir}/result.png"
    --stats-output "${out_dir}/stats.json"
    --metadata-output "${out_dir}/metadata.json"
    --n-max "${n_max}"
    --src-guidance-scale "${src_cfg}"
    --tar-guidance-scale "${src_cfg}"
    --edit-hedit-guidance-scale 0
    --rec-guidance-scale 0
    --edit-guidance-scale 0
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

run_case nmax28_cfg35 28 3.5
run_case nmax24_cfg10 24 1.0
run_case nmax28_cfg10 28 1.0
