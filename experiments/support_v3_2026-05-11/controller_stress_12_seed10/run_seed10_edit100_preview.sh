#!/usr/bin/env bash
set -euo pipefail

cd /home/Wu_25R8111/rf_h_edit_project

PARALLEL=1 \
GPU_DEVICES="6 7" \
MAX_PARALLEL_JOBS=2 \
RUN_EDIT_STRENGTH=1 \
RUN_SUPPORT_PERTURB=0 \
RUN_ANALYZE=0 \
TASKS="P1 P2 P3 P5 P6 P7 P8 P10 P11 P12 P13" \
METHODS="M17 M18" \
SEEDS="10" \
EDIT_SCALES="1.0" \
SKIP_EXISTING=1 \
REGENERATE_MASKS=0 \
ALLOW_MASK_DOWNLOAD=0 \
LOG_DIR="/home/Wu_25R8111/rf_h_edit_project/experiments/support_v3_2026-05-11/controller_stress_12_seed10/logs_edit100_preview" \
ANALYSIS_DIR="/home/Wu_25R8111/rf_h_edit_project/experiments/support_v3_2026-05-11/controller_stress_12_seed10" \
scripts/run_controller_stress_sweeps.sh
