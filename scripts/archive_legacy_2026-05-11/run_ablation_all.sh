#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
OUT_DIR="${ROOT}/outputs/ablation_all"
mkdir -p "${OUT_DIR}"

printf '%q ' "$0" "$@" > "${OUT_DIR}/command.txt"
printf '\n' >> "${OUT_DIR}/command.txt"

"${ROOT}/scripts/run_base_only.sh"
"${ROOT}/scripts/run_direct_target.sh"
"${ROOT}/scripts/run_decoupled_rec.sh"
"${ROOT}/scripts/run_anchor_only.sh"
"${ROOT}/scripts/run_sunglasses_local.sh"
if [[ "${RUN_EXTERNAL_MASK:-0}" == "1" ]]; then
  "${ROOT}/scripts/run_sunglasses_external_mask.sh"
fi

cat > "${OUT_DIR}/metadata.json" <<JSON
{
  "description": "Runs the core RF h-Edit ablation scripts. External-mask diagnostics are opt-in with RUN_EXTERNAL_MASK=1.",
  "run_external_mask": "${RUN_EXTERNAL_MASK:-0}"
}
JSON
