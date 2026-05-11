#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
PYTHON="${PYTHON:-/home/Wu_25R8111/ENTER/envs/flowedit/bin/python}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/paper_figures}"
SEEDS="${SEEDS:-10}"
TASKS="${TASKS:-cat_crown backpack_blue yellow_car_blue rabbit_sunglasses red_chair_blue}"
METHODS="${METHODS:-base_only direct_target anchor_only decoupled_rec full}"

mkdir -p "${OUT_ROOT}"

for task in ${TASKS}; do
  for seed in ${SEEDS}; do
    items=()
    for method in ${METHODS}; do
      result="${ROOT}/outputs/main_matrix/${task}/${method}/seed_${seed}/result.png"
      if [[ -s "${result}" ]]; then
        items+=("${method}=${result}")
      fi
    done
    if (( ${#items[@]} == 0 )); then
      echo "[paper-figures] skip task=${task} seed=${seed}: no result images"
      continue
    fi
    output="${OUT_ROOT}/${task}_seed_${seed}.png"
    echo "[paper-figures] ${output}"
    "${PYTHON}" "${ROOT}/scripts/make_comparison_grid.py" --output "${output}" "${items[@]}"
  done
done
