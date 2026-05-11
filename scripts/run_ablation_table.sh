#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/Wu_25R8111/rf_h_edit_project}"
RUNNER="${ROOT}/scripts/run_main_table.sh"
TASKS="${TASKS:-T1 T2 T3}"
SEEDS="${SEEDS:-10}"
DEVICE="${DEVICE:-0}"

run_variant() {
  local name="$1"
  shift
  echo "[ablation] ${name}"
  env \
    ROOT="${ROOT}" \
    TASKS="${TASKS}" \
    METHODS="M4" \
    SEEDS="${SEEDS}" \
    DEVICE="${DEVICE}" \
    METHOD_NAME_SUFFIX="${name}" \
    "$@" \
    "${RUNNER}"
}

run_variant "full_ref" \
  SOURCE_INJECT_V_SCALE="${SOURCE_INJECT_V_SCALE:-0.0}"

run_variant "no_rec" \
  FORCE_REC_GUIDANCE_SCALE="0.0"

run_variant "no_traj" \
  FORCE_TRAJECTORY_PRESERVE_SCALE="0.0"

run_variant "attention_velocity" \
  FORCE_OBJECT_MASK_PROVIDER="attention_velocity" \
  SURFACE_REF_MODE="off"

run_variant "semantic_velocity" \
  FORCE_OBJECT_MASK_PROVIDER="semantic_velocity"

run_variant "source_v_inject" \
  SOURCE_INJECT_V_SCALE="${SOURCE_INJECT_V_SCALE:-0.35}" \
  SOURCE_INJECT_MASK_MODE="${SOURCE_INJECT_MASK_MODE:-edit}" \
  SOURCE_INJECT_STEPS="${SOURCE_INJECT_STEPS:-8}"
