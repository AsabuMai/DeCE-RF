#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
COMPARE_ROOT="${COMPARE_ROOT:-/home/Wu_25R8111/rf_editing_compare_sunglasses}"
HEDIT_RESULT="${HEDIT_RESULT:-/home/Wu_25R8111/h-edit/text-guided/results/panda_sunglasses_demo_try/h_edit_R_p2p_total_steps_20_skip_0_implicit_True_eta_1.0_src_orig_1.0_src_edit_5.0_tar_scale_7.5_w_rec_0.1_n_opts_1_time_1776754298_xa_0.4_sa0.35_/panda.jpg}"
OUT_DIR="${ROOT}/outputs/cross_method_sunglasses_masks"

mkdir -p "${OUT_DIR}"

"${PYTHON}" "${ROOT}/scripts/make_comparison_grid.py" \
  --output "${ROOT}/outputs/sunglasses_cross_method_grid.png" \
  source="${ROOT}/h_edit_compare/panda.jpg" \
  ours="${ROOT}/outputs/sunglasses_local/result.png" \
  ours_mask="${ROOT}/outputs/sunglasses_local/masks/subject_final.png" \
  rf_solver="${COMPARE_ROOT}/outputs/rf_solver/img_0.jpg" \
  fireflow="${COMPARE_ROOT}/outputs/fireflow/panda_sunglasses_inject_1_start_layer_index_0_end_layer_index_37_img_0.jpg" \
  rf_inv="${COMPARE_ROOT}/outputs/rf_inversion/panda_sunglasses_rf_inversion.png" \
  h_edit="${HEDIT_RESULT}"

for method in ours rf_solver fireflow; do
  case "${method}" in
    ours)
      proposal="${ROOT}/outputs/sunglasses_local/result.png"
      ;;
    rf_solver)
      proposal="${COMPARE_ROOT}/outputs/rf_solver/img_0.jpg"
      ;;
    fireflow)
      proposal="${COMPARE_ROOT}/outputs/fireflow/panda_sunglasses_inject_1_start_layer_index_0_end_layer_index_37_img_0.jpg"
      ;;
  esac
  "${PYTHON}" "${ROOT}/scripts/proposal_local_composite.py" \
    --source "${ROOT}/h_edit_compare/panda.jpg" \
    --proposal "${proposal}" \
    --output-dir "${OUT_DIR}" \
    --name "${method}" \
    --roi 0.20,0.15,0.78,0.60 \
    --threshold 0.12 \
    --keep-components 5 \
    --min-area 8 \
    --dilate 5 \
    --blur 11 \
    --dark-bias 2.0 \
    --mode alpha
done

"${PYTHON}" "${ROOT}/scripts/make_comparison_grid.py" \
  --output "${ROOT}/outputs/sunglasses_cross_method_mask_grid.png" \
  source="${ROOT}/h_edit_compare/panda.jpg" \
  ours="${ROOT}/outputs/sunglasses_local/result.png" \
  ours_attn_mask="${ROOT}/outputs/sunglasses_local/masks/subject_final.png" \
  ours_diff="${OUT_DIR}/ours_mask.png" \
  rf_solver="${COMPARE_ROOT}/outputs/rf_solver/img_0.jpg" \
  solver_diff="${OUT_DIR}/rf_solver_mask.png" \
  fireflow="${COMPARE_ROOT}/outputs/fireflow/panda_sunglasses_inject_1_start_layer_index_0_end_layer_index_37_img_0.jpg" \
  fireflow_diff="${OUT_DIR}/fireflow_mask.png"
