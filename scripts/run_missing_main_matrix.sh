#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
DEVICE="${DEVICE:-0}"
TASKS="${TASKS:-T1 T2 T3 T4}"
METHODS="${METHODS:-M0 M1 M2 M3 M4}"
SEEDS="${SEEDS:-10 11 12}"
LIMIT="${LIMIT:-1}"
DRY_RUN="${DRY_RUN:-0}"
AUDIT_JSON="${AUDIT_JSON:-${ROOT}/experiments/main_matrix_coverage_audit.json}"

"${PYTHON}" "${ROOT}/scripts/audit_main_matrix_coverage.py" \
  --outputs-dir "${ROOT}/outputs/main_matrix" \
  --tasks "${TASKS}" \
  --methods "${METHODS}" \
  --seeds "${SEEDS}" \
  --json-output "${AUDIT_JSON}" || true

mapfile -t missing_cells < <(
  "${PYTHON}" - "${AUDIT_JSON}" <<'PY'
import json
import sys

records = json.load(open(sys.argv[1], encoding="utf-8"))
for record in records:
    if not record.get("complete"):
        print(record["task_id"], record["method_id"], record["seed"])
PY
)

count=0
for cell in "${missing_cells[@]}"; do
  read -r task method seed <<<"${cell}"
  if (( LIMIT > 0 && count >= LIMIT )); then
    break
  fi
  count=$((count + 1))
  echo "[missing-main-matrix] ${count}: task=${task} method=${method} seed=${seed}"
  TASKS="${task}" METHODS="${method}" SEEDS="${seed}" DEVICE="${DEVICE}" \
    ROOT="${ROOT}" PYTHON="${PYTHON}" DRY_RUN="${DRY_RUN}" SKIP_EXISTING=1 \
    bash "${ROOT}/scripts/run_main_table.sh"
done

echo "[missing-main-matrix] attempted ${count} missing cells"
