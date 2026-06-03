#!/usr/bin/env bash
set -euo pipefail

cd /home/Wu_25R8111/rf_h_edit_project

PARALLEL=1 \
GPU_DEVICES="6 7" \
MAX_PARALLEL_JOBS=2 \
RUN_EDIT_STRENGTH=1 \
RUN_SUPPORT_PERTURB=1 \
RUN_ANALYZE=1 \
COMPUTE_LPIPS=1 \
TASKS="P1 P2 P3 P5 P6 P7 P8 P10 P11 P12 P13" \
ANALYSIS_TASKS="cat_crown dog_sunglasses mug_heart tshirt_star tote_leaf backpack_remove_toy_charm backpack_replace_patch_blue cat_replace_bell_heart_tag dog_replace_tennis_ball_star rabbit_sunglasses dog_crown" \
METHODS="M17 M18" \
SEEDS="10" \
EDIT_SCALES="0.5 0.75 1.0 1.25 1.5 2.0" \
PERTURBATIONS="erode dilate shift boundary_noise holes" \
SKIP_EXISTING=1 \
REGENERATE_MASKS=0 \
ALLOW_MASK_DOWNLOAD=0 \
ALLOW_METRIC_DOWNLOAD=0 \
CLIP_DEVICE="cuda:6" \
ANALYSIS_DIR="/home/Wu_25R8111/rf_h_edit_project/experiments/support_v3_2026-05-11/controller_stress_12_seed10" \
scripts/run_controller_stress_sweeps.sh
