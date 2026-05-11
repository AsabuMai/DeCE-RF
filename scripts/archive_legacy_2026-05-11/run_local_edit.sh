#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"

export OUT_DIR="${OUT_DIR:-${ROOT}/outputs/local_edit}"

"${ROOT}/scripts/run_sunglasses_local.sh"
