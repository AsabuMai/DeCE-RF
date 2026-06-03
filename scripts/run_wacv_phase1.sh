#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUNNER="${RUNNER:-${ROOT}/scripts/run_pretty_matrix.sh}"

SCOPE="${SCOPE:-implemented}"
METHODS="${METHODS:-base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap}"
SEEDS="${SEEDS:-10 11 12}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

STRICT_TASKS="cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_vertical_fabric_strip backpack_remove_toy_charm"
IMPLEMENTED_TASKS="cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_vertical_fabric_strip backpack_remove_toy_charm"
OLD_SERVER_EVIDENCE_TASKS="cat_crown dog_sunglasses mug_heart tshirt_star backpack_remove_toy_charm red_chair_blue"
PENDING_STRICT_TASKS="${PENDING_STRICT_TASKS:-}"

usage() {
  cat <<EOF
Usage: SCOPE=<implemented|strict|old_server_evidence> [METHODS=...] [SEEDS=...] bash scripts/run_wacv_phase1.sh

Scopes:
  implemented          Runs the updated WACV tasks currently implemented in scripts/run_pretty_matrix.sh.
  strict               Runs the updated strict Core-6 task set; set PENDING_STRICT_TASKS to re-enable the missing-task guard.
  old_server_evidence  Replays the previous server-evidence task set for diagnostics only.

Default METHODS: ${METHODS}
Default SEEDS:   ${SEEDS}
EOF
}

case "${SCOPE}" in
  implemented)
    TASKS="${TASKS:-${IMPLEMENTED_TASKS}}"
    ;;
  strict)
    if [[ -n "${PENDING_STRICT_TASKS}" && "${ALLOW_PENDING_TASKS:-0}" != "1" ]]; then
      echo "[wacv-phase1] strict Core-6 is not runnable yet." >&2
      echo "[wacv-phase1] pending task definitions: ${PENDING_STRICT_TASKS}" >&2
      echo "[wacv-phase1] use SCOPE=implemented for the runnable subset, or add the missing task definitions first." >&2
      exit 2
    fi
    TASKS="${TASKS:-${STRICT_TASKS}}"
    ;;
  old_server_evidence)
    TASKS="${TASKS:-${OLD_SERVER_EVIDENCE_TASKS}}"
    echo "[wacv-phase1] WARNING: old_server_evidence is diagnostic only; it does not complete updated T2/T5." >&2
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "[wacv-phase1] unknown SCOPE='${SCOPE}'" >&2
    usage >&2
    exit 2
    ;;
esac

echo "[wacv-phase1] scope=${SCOPE}"
echo "[wacv-phase1] tasks=${TASKS}"
echo "[wacv-phase1] methods=${METHODS}"
echo "[wacv-phase1] seeds=${SEEDS}"
echo "[wacv-phase1] skip_existing=${SKIP_EXISTING} dry_run=${DRY_RUN}"

TASKS="${TASKS}" \
METHODS="${METHODS}" \
SEEDS="${SEEDS}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
DRY_RUN="${DRY_RUN}" \
bash "${RUNNER}"
