#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
FORCE="${FORCE:-0}"

run_case() {
  local name="$1"
  shift

  local out_dir="${ROOT}/outputs/panda_tiger_diag_${name}"
  mkdir -p "${out_dir}/masks"
  if [[ "${FORCE}" != "1" && -f "${out_dir}/result.png" && -f "${out_dir}/stats.json" ]]; then
    echo "[skip] panda_tiger_diag_${name} already exists"
    return
  fi

  local cmd=(
    "${PYTHON}" "${ROOT}/run_edit_sd3.py"
    --image "${ROOT}/h_edit_compare/panda.jpg"
    --source-prompt "A panda is walking in a forest."
    --prompt "A tiger is walking in a forest."
    --output "${out_dir}/result.png"
    --stats-output "${out_dir}/stats.json"
    --metadata-output "${out_dir}/metadata.json"
    --mask-output-dir "${out_dir}/masks"
    --src-guidance-scale 1.0
    --tar-guidance-scale 10.5
    --rec-guidance-scale 0
    --edit-hedit-guidance-scale 0
    --edit-guidance-scale 0
    --edit-region-guidance-scale 0
    --edit-target-guidance-scale 0
    --edit-source-guidance-scale 0
    --edit-clip-guidance-scale 0
    --edit-text-guidance-scale 0
    --edit-dds-guidance-scale 0
    --edit-app-guidance-scale 0
    --attention-mask-mode changed_union
    --attention-mask-subject-threshold 0.48
    --attention-mask-core-threshold 0.72
    --beta-max 1.0
    --velocity-conversion-mode linear_path
    --linear-path-t-min 0.05
    --rec-stop-timestep 0.08
    --photo-prompt-mode both
    --log-every 7
    "$@"
  )

  printf '%q ' "CUDA_VISIBLE_DEVICES=${DEVICE}" "${cmd[@]}" > "${out_dir}/command.txt"
  printf '\n' >> "${out_dir}/command.txt"
  CUDA_VISIBLE_DEVICES="${DEVICE}" "${cmd[@]}"
}

run_case flow_only \
  --edit-hedit-guidance-scale 1.0

run_case clean_anchor \
  --edit-guidance-scale 1.0

run_case region_anchor \
  --edit-region-guidance-scale 1.0

run_case flow_plus_traj \
  --edit-hedit-guidance-scale 1.0 \
  --trajectory-preserve-scale 0.15

run_case target_feature \
  --edit-target-guidance-scale 0.5

run_case source_suppress \
  --edit-source-guidance-scale 0.5
