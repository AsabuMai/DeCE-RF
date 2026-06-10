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
RUN_PRETTY_CONFIG_SOURCE="${RUN_PRETTY_CONFIG_SOURCE:-core6}"

# Strict Core-5 (revised 2026-06-10): removal moved to the E5 boundary probe
# because decoupled clean-displacement control has no defined edit target for
# fill content; run backpack_remove_toy_charm via E5_REMOVAL_PROBE_TASKS.
STRICT_TASKS="cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_same_color_cable_knit"
IMPLEMENTED_TASKS="cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_same_color_cable_knit"
E5_REMOVAL_PROBE_TASKS="backpack_remove_toy_charm"
OLD_SERVER_EVIDENCE_TASKS="cat_crown dog_sunglasses mug_heart tshirt_star backpack_remove_toy_charm red_chair_blue"
PENDING_STRICT_TASKS="${PENDING_STRICT_TASKS:-}"

usage() {
  cat <<EOF
Usage: SCOPE=<implemented|strict|e5_removal_probe|old_server_evidence> [METHODS=...] [SEEDS=...] bash scripts/run_wacv_phase1.sh

Scopes:
  implemented          Runs the updated WACV tasks currently implemented in scripts/run_pretty_matrix.sh.
  strict               Runs the strict Core-5 task set (removal lives in the E5 boundary probe); set PENDING_STRICT_TASKS to re-enable the missing-task guard.
  e5_removal_probe     Runs the E5 removal boundary probe task(s).
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
      echo "[wacv-phase1] strict Core-5 is not runnable yet." >&2
      echo "[wacv-phase1] pending task definitions: ${PENDING_STRICT_TASKS}" >&2
      echo "[wacv-phase1] use SCOPE=implemented for the runnable subset, or add the missing task definitions first." >&2
      exit 2
    fi
    TASKS="${TASKS:-${STRICT_TASKS}}"
    ;;
  e5_removal_probe)
    TASKS="${TASKS:-${E5_REMOVAL_PROBE_TASKS}}"
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
echo "[wacv-phase1] runner_config_source=${RUN_PRETTY_CONFIG_SOURCE}"

TASKS="${TASKS}" \
METHODS="${METHODS}" \
SEEDS="${SEEDS}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
DRY_RUN="${DRY_RUN}" \
RUN_PRETTY_CONFIG_SOURCE="${RUN_PRETTY_CONFIG_SOURCE}" \
bash "${RUNNER}"
