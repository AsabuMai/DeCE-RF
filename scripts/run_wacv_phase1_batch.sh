#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
RUN_ID="${RUN_ID:-wacv_phase1_batch_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/${RUN_ID}}"
MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/batch_manifest.txt}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${OUTPUT_ROOT}/batch_summary.json}"

mkdir -p "${OUTPUT_ROOT}"
rm -f "${MANIFEST}"

echo "[wacv-phase1-batch] run_id=${RUN_ID}"
echo "[wacv-phase1-batch] output_root=${OUTPUT_ROOT}"
echo "[wacv-phase1-batch] manifest=${MANIFEST}"
echo "[wacv-phase1-batch] summary=${SUMMARY_OUTPUT}"

OUTPUT_ROOT="${OUTPUT_ROOT}" \
BATCH_MANIFEST="${MANIFEST}" \
MODEL_OFFLOAD="${MODEL_OFFLOAD:-0}" \
SCOPE="${SCOPE:-strict}" \
METHODS="${METHODS:-base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap}" \
SEEDS="${SEEDS:-10}" \
SKIP_EXISTING="${SKIP_EXISTING:-0}" \
DRY_RUN="${DRY_RUN:-0}" \
bash "${ROOT}/scripts/run_wacv_phase1.sh"

count="$(wc -l < "${MANIFEST}" | tr -d ' ')"
echo "[wacv-phase1-batch] queued=${count}"
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

batch_args=(
  "--manifest" "${MANIFEST}"
  "--summary-output" "${SUMMARY_OUTPUT}"
)
if [[ "${BATCH_SKIP_EXISTING:-0}" == "1" ]]; then
  batch_args+=("--skip-existing")
fi

"${PYTHON}" "${ROOT}/scripts/run_sd3_batch.py" "${batch_args[@]}"
