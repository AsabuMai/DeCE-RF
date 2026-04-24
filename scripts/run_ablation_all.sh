#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"

"${ROOT}/scripts/run_base_only.sh"
"${ROOT}/scripts/run_direct_target.sh"
"${ROOT}/scripts/run_decoupled_rec.sh"
"${ROOT}/scripts/run_anchor_only.sh"
