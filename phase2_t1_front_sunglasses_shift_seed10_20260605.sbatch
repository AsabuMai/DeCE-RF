#!/usr/bin/env bash
#SBATCH -J phase2_t1_face_shift
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 01:30:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t1_front_sunglasses_shift_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_t1_front_sunglasses_shift_seed10_20260605_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
ARTIFACT="${RUN_DIR}/t1_front_sunglasses_shift_${SLURM_JOB_ID:-manual}.tgz"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[t1-front-shift] start=$(date -Is)"
echo "[t1-front-shift] host=$(hostname)"
echo "[t1-front-shift] job_id=${SLURM_JOB_ID:-none}"
echo "[t1-front-shift] run_dir=${RUN_DIR}"
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

prepare_variant() {
  local suffix="$1"
  local shift_y="$2"
  local max_area="$3"
  local dilate="$4"
  local hedit="$5"
  local text="$6"
  local rec="$7"
  local traj="$8"
  local drift="$9"

  METHOD_NAME_SUFFIX="${suffix}" \
  EDIT_MASK_SHIFT_Y="${shift_y}" \
  SUPPORT_V3_FACE_MIN_AREA_RATIO=0.020 \
  SUPPORT_V3_FACE_MAX_AREA_RATIO="${max_area}" \
  SUPPORT_V3_FACE_DILATE_RADIUS="${dilate}" \
  SUPPORT_V3_FACE_BLUR_KERNEL=3 \
  SUPPORT_V3_FACE_HEDIT="${hedit}" \
  SUPPORT_V3_FACE_TEXT="${text}" \
  SUPPORT_V3_FACE_REC_GUIDANCE_SCALE="${rec}" \
  SUPPORT_V3_FACE_TRAJECTORY_PRESERVE_SCALE="${traj}" \
  SUPPORT_V3_FACE_PRESERVE_DRIFT_BUDGET="${drift}" \
  SUPPORT_V3_FACE_OUTSIDE_LOCK=0.88 \
  BATCH_MANIFEST="${MANIFEST}" \
  PREPARE_ONLY=1 \
  SKIP_EXISTING=0 \
  TASKS="dog_front_sunglasses_phase2" \
  METHODS="support_v3_controller_rmsgap" \
  SEEDS="10" \
  bash scripts/run_pretty_matrix.sh
}

prepare_variant "_frontS_shift055" 0.055 0.110 3 0.68 0.10 0.20 0.10 0.17
prepare_variant "_frontT_shift085" 0.085 0.110 3 0.68 0.10 0.20 0.10 0.17
prepare_variant "_frontU_shift085_strong" 0.085 0.130 4 0.76 0.14 0.18 0.08 0.18

echo "[t1-front-shift] manifest_count=$(wc -l < "${MANIFEST}")"

"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
methods = [
    "support_v3_controller_rmsgap_frontS_shift055",
    "support_v3_controller_rmsgap_frontT_shift085",
    "support_v3_controller_rmsgap_frontU_shift085_strong",
]
missing = {}
complete = 0
for method in methods:
    d = root / "outputs" / "pretty_matrix" / "dog_front_sunglasses_phase2" / method / "seed_10"
    miss = [name for name in ("result.png", "metadata.json", "stats.json", "command.txt") if not (d / name).is_file()]
    if miss:
        missing[method] = miss
    else:
        complete += 1
summary = {"complete": complete, "expected": len(methods), "missing": missing}
run_dirs = sorted((root / "experiments" / "support_v3_2026-06-02").glob("phase2_t1_front_sunglasses_shift_seed10_20260605_*"))
summary_path = run_dirs[-1] / "phase2_t1_front_sunglasses_shift_completion.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[t1-front-shift] complete={complete}")
print(f"[t1-front-shift] expected={len(methods)}")
print(f"[t1-front-shift] missing={len(missing)}")
if missing:
    raise SystemExit(1)
PY

tar -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  "experiments/support_v3_2026-06-02/${RUN_ID}" \
  "outputs/pretty_matrix/dog_front_sunglasses_phase2/support_v3_controller_rmsgap_frontS_shift055" \
  "outputs/pretty_matrix/dog_front_sunglasses_phase2/support_v3_controller_rmsgap_frontT_shift085" \
  "outputs/pretty_matrix/dog_front_sunglasses_phase2/support_v3_controller_rmsgap_frontU_shift085_strong" \
  "data/pretty_free_candidates/commons_dog_front_medusa.jpg"

echo "[t1-front-shift] artifact=${ARTIFACT}"
echo "[t1-front-shift] end=$(date -Is)"
