#!/usr/bin/env bash
set -euo pipefail

cd /home/Wu_25R8111/rf_h_edit_project

for noise in 0.08 0.16; do
  tag="$(/home/Wu_25R8111/ENTER/envs/flowedit/bin/python - "$noise" <<'PY'
import sys
print(f"{int(round(float(sys.argv[1]) * 100)):03d}")
PY
)"
  DEVICE=7 \
  TASKS="P5 P6" \
  METHODS="M17" \
  SEEDS="10" \
  METHOD_NAME_SUFFIX="_rfrefresh${tag}" \
  EDIT_INITIAL_NOISE_SCALE="${noise}" \
  EDIT_INITIAL_NOISE_REGION="core" \
  SKIP_EXISTING=1 \
  REGENERATE_MASKS=0 \
  ALLOW_MASK_DOWNLOAD=0 \
  scripts/run_pretty_matrix.sh
done
