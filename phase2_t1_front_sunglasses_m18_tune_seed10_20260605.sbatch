#!/usr/bin/env bash
#SBATCH -J phase2_t1_m18face
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 01:30:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t1_front_sunglasses_m18_tune_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_t1_front_sunglasses_m18_tune_seed10_20260605_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
ARTIFACT="${RUN_DIR}/t1_front_sunglasses_m18_tune_${SLURM_JOB_ID:-manual}.tgz"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[t1-front-m18] start=$(date -Is)"
echo "[t1-front-m18] host=$(hostname)"
echo "[t1-front-m18] job_id=${SLURM_JOB_ID:-none}"
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

PROMPT="A close-up front-facing portrait of the same dog wearing clearly visible opaque black sunglasses with dark lenses aligned across both eyes indoors, while the dog face, ears, fur, nose, floor, and lighting remain unchanged."
TASKS_ONE="dog_front_sunglasses_phase2"
METHODS_ONE="support_v3_controller_rmsgap"
SEEDS_ONE="10"

prepare_one() {
  local suffix="$1"
  local candidate="$2"
  shift 2
  echo "[t1-front-m18] preparing ${suffix}"
  env "$@" \
    METHOD_NAME_SUFFIX="${suffix}" \
    BATCH_MANIFEST="${MANIFEST}" \
    PREPARE_ONLY=1 \
    SKIP_EXISTING=0 \
    TARGET_PROMPT_OVERRIDE="${PROMPT}" \
    ATTENTION_TARGET_WORDS_OVERRIDE="opaque,black,sunglasses,dark,lenses,eyes" \
    CHANGED_TARGET_WORDS_OVERRIDE="sunglasses,lenses" \
    SUPPORT_NEW_TOKENS_OVERRIDE="sunglasses,dark lenses" \
    SUPPORT_HOST_TOKENS_OVERRIDE="dog head,eyes" \
    SUPPORT_V3_CANDIDATE_OVERRIDE="${candidate}" \
    TASKS="${TASKS_ONE}" \
    METHODS="${METHODS_ONE}" \
    SEEDS="${SEEDS_ONE}" \
    bash scripts/run_pretty_matrix.sh
}

prepare_one "_frontQ_m18_black_response" "relation_x_response" \
  SUPPORT_V3_FACE_HEDIT=0.86 \
  SUPPORT_V3_FACE_TEXT=0.20 \
  SUPPORT_V3_FACE_ANCHOR=0.04 \
  SUPPORT_V3_FACE_REGION=0.06 \
  SUPPORT_V3_FACE_REC_GUIDANCE_SCALE=0.16 \
  SUPPORT_V3_FACE_TRAJECTORY_PRESERVE_SCALE=0.08 \
  SUPPORT_V3_FACE_PRESERVE_DRIFT_BUDGET=0.16 \
  SUPPORT_V3_FACE_OUTSIDE_LOCK=0.86 \
  SUPPORT_V3_FACE_MIN_AREA_RATIO=0.020 \
  SUPPORT_V3_FACE_MAX_AREA_RATIO=0.110 \
  SUPPORT_V3_FACE_DILATE_RADIUS=3 \
  SUPPORT_V3_FACE_BLUR_KERNEL=3

prepare_one "_frontR_m18_black_relonly" "relation_only" \
  SUPPORT_V3_FACE_HEDIT=0.78 \
  SUPPORT_V3_FACE_TEXT=0.18 \
  SUPPORT_V3_FACE_ANCHOR=0.03 \
  SUPPORT_V3_FACE_REGION=0.05 \
  SUPPORT_V3_FACE_REC_GUIDANCE_SCALE=0.18 \
  SUPPORT_V3_FACE_TRAJECTORY_PRESERVE_SCALE=0.10 \
  SUPPORT_V3_FACE_PRESERVE_DRIFT_BUDGET=0.15 \
  SUPPORT_V3_FACE_OUTSIDE_LOCK=0.88 \
  SUPPORT_V3_FACE_MIN_AREA_RATIO=0.020 \
  SUPPORT_V3_FACE_MAX_AREA_RATIO=0.110 \
  SUPPORT_V3_FACE_DILATE_RADIUS=3 \
  SUPPORT_V3_FACE_BLUR_KERNEL=3

echo "[t1-front-m18] manifest_count=$(wc -l < "${MANIFEST}")"

"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
methods = [
    "support_v3_controller_rmsgap_frontQ_m18_black_response",
    "support_v3_controller_rmsgap_frontR_m18_black_relonly",
]
missing = []
for method in methods:
    d = root / "outputs" / "pretty_matrix" / "dog_front_sunglasses_phase2" / method / "seed_10"
    absent = [name for name in ("result.png", "metadata.json", "stats.json", "command.txt") if not (d / name).is_file()]
    if absent:
        missing.append({"method": method, "missing": absent})
summary = {"complete": len(methods) - len(missing), "expected": len(methods), "missing_count": len(missing), "missing": missing}
run_dirs = sorted((root / "experiments" / "support_v3_2026-06-02").glob("phase2_t1_front_sunglasses_m18_tune_seed10_20260605_*"))
run_dir = run_dirs[-1]
(run_dir / "phase2_t1_front_sunglasses_m18_tune_seed10_completion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[t1-front-m18] complete={summary['complete']}")
print(f"[t1-front-m18] expected={summary['expected']}")
print(f"[t1-front-m18] missing={summary['missing_count']}")
if missing:
    raise SystemExit(1)
PY

tar -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  "experiments/support_v3_2026-06-02/${RUN_ID}" \
  "outputs/pretty_matrix/dog_front_sunglasses_phase2/support_v3_controller_rmsgap_frontQ_m18_black_response" \
  "outputs/pretty_matrix/dog_front_sunglasses_phase2/support_v3_controller_rmsgap_frontR_m18_black_relonly" \
  "data/pretty_free_candidates/commons_dog_front_medusa.jpg"

echo "[t1-front-m18] artifact=${ARTIFACT}"
echo "[t1-front-m18] end=$(date -Is)"
