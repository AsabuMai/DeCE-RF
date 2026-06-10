#!/usr/bin/env bash
#SBATCH -J phase2_t3_mugc
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 01:30:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t3_mug_cdelta_repair_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_t3_mug_cdelta_repair_seed10_20260605_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
ARTIFACT="${ROOT}/experiments/support_v3_2026-06-02/phase2_t3_mug_cdelta_repair_seed10_${SLURM_JOB_ID:-manual}.tgz"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[t3-mug-cdelta] start=$(date -Is)"
echo "[t3-mug-cdelta] host=$(hostname)"
echo "[t3-mug-cdelta] job_id=${SLURM_JOB_ID:-none}"
nvidia-smi || true

export PYTHON="${ROOT}/.venv/bin/python"
export HF_HOME="${ROOT}/../.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MODEL_OFFLOAD=0
export LOW_VRAM=0
export NUM_INFERENCE_STEPS=28
export N_MAX=24
export MAX_IMAGE_SIZE=512
export REGENERATE_MASKS=1
export REUSE_SEMANTIC_MASKS=0

PROMPT="A minimalist photo of the same white ceramic mug with a small flat hard-edged solid red heart decal printed cleanly on the front, with sharp printed edges and no glow or blur, while the mug shape, handle, highlights, bottom shadow, grey background, and lighting remain unchanged."

add_variant() {
  local suffix="$1"
  local cdelta="$2"
  local ref="$3"
  BATCH_MANIFEST="${MANIFEST}" \
  PREPARE_ONLY=1 \
  SKIP_EXISTING=0 \
  TASKS="mug_heart" \
  METHODS="support_v3_controller_rmsgap" \
  SEEDS="10" \
  METHOD_NAME_SUFFIX="${suffix}" \
  TARGET_PROMPT_OVERRIDE="${PROMPT}" \
  SUPPORT_V3_LOCALIZED_DECAL_REF_GUIDANCE_SCALE="${ref}" \
  SUPPORT_V3_LOCALIZED_DECAL_REF_LUMA_PRESERVE="0.00" \
  SUPPORT_V3_LOCALIZED_DECAL_REF_GRADIENT_PRESERVE="0.00" \
  SUPPORT_V3_LOCALIZED_DECAL_REF_SMOOTH_KERNEL=0 \
  SUPPORT_V3_OPFIELD_DECAL_REC="0.16" \
  SUPPORT_V3_OPFIELD_DECAL_TRAJ="0.08" \
  SUPPORT_V3_OPFIELD_DECAL_HEDIT="0.68" \
  SUPPORT_V3_OPFIELD_DECAL_TEXT="0.12" \
  COMPLETION_CLEAN_DELTA_SCALE="${cdelta}" \
  COMPLETION_CLEAN_DELTA_SCHEDULE_START="0.0" \
  COMPLETION_CLEAN_DELTA_SCHEDULE_STOP="0.0" \
  bash scripts/run_pretty_matrix.sh
}

add_variant "_mugcdelta025" "0.25" "0.70"
add_variant "_mugcdelta045" "0.45" "0.65"
add_variant "_mugcdelta070" "0.70" "0.55"

echo "[t3-mug-cdelta] manifest_count=$(wc -l < "${MANIFEST}")"

"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}"

"${PYTHON}" - <<'PY'
from pathlib import Path
root = Path("/cluster/users/grad/2025/25t8103/project")
suffixes = ["_mugcdelta025", "_mugcdelta045", "_mugcdelta070"]
missing = []
for suffix in suffixes:
    d = root / "outputs" / "pretty_matrix" / "mug_heart" / ("support_v3_controller_rmsgap" + suffix) / "seed_10"
    absent = [name for name in ("result.png", "metadata.json", "stats.json", "command.txt") if not (d / name).is_file()]
    if absent:
        missing.append({"suffix": suffix, "missing": absent})
print(f"[t3-mug-cdelta] missing={missing}")
if missing:
    raise SystemExit(1)
PY

tar -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  "experiments/support_v3_2026-06-02/${RUN_ID}" \
  "outputs/pretty_matrix/mug_heart/support_v3_controller_rmsgap_mugcdelta025" \
  "outputs/pretty_matrix/mug_heart/support_v3_controller_rmsgap_mugcdelta045" \
  "outputs/pretty_matrix/mug_heart/support_v3_controller_rmsgap_mugcdelta070"

echo "[t3-mug-cdelta] artifact=${ARTIFACT}"
echo "[t3-mug-cdelta] end=$(date -Is)"
