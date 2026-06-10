#!/usr/bin/env bash
#SBATCH -J phase2_prof_tune
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 02:30:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_profile_face_tune_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_profile_face_tune_seed10_20260605_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
ARTIFACT="${RUN_DIR}/profile_face_tune_${SLURM_JOB_ID:-manual}.tgz"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[profile-tune] start=$(date -Is)"
echo "[profile-tune] host=$(hostname)"
echo "[profile-tune] cwd=$(pwd)"
echo "[profile-tune] job_id=${SLURM_JOB_ID:-none}"
echo "[profile-tune] run_dir=${RUN_DIR}"
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

TASKS_ONE="cat_side_sunglasses_phase2"
METHODS_ONE="support_v3_controller_rmsgap"
SEEDS_ONE="10"
STRICT_PROMPT="A strict side-profile photo of the same cat facing right in the same green grass, with only the visible eye area edited to have a small black side-view sunglass lens over its visible eye, while the head direction, one-eye-visible profile, fur outline, pose, grass, and lighting remain unchanged."
OVAL_PROMPT="A strict side-profile photo of the same cat facing right in the same green grass, with only the single visible eye edited to have a small black oval side-view eyepiece lens, while the head direction, one-eye-visible profile, fur outline, pose, grass, and lighting remain unchanged."
STRICT_LOCAL="A close-up strict side-profile photo of one small black side-view sunglass lens over the single visible eye of a cat."
OVAL_LOCAL="A close-up strict side-profile photo of a small black oval eyepiece lens over the single visible eye of a cat."

prepare_profile() {
  local suffix="$1"
  local prompt="$2"
  local local_prompt="$3"
  local attn_words="$4"
  local new_tokens="$5"
  shift 5
  echo "[profile-tune] preparing ${suffix}"
  env "$@" \
    METHOD_NAME_SUFFIX="${suffix}" \
    BATCH_MANIFEST="${MANIFEST}" \
    PREPARE_ONLY=1 \
    SKIP_EXISTING=0 \
    TARGET_PROMPT_OVERRIDE="${prompt}" \
    ATTENTION_TARGET_WORDS_OVERRIDE="${attn_words}" \
    CHANGED_TARGET_WORDS_OVERRIDE="${new_tokens}" \
    SUPPORT_NEW_TOKENS_OVERRIDE="${new_tokens}" \
    SUPPORT_HOST_TOKENS_OVERRIDE="cat head,visible eye" \
    SUPPORT_V3_RELATION_OVERRIDE="on_profile_face_right" \
    SEMANTIC_PHRASE_OVERRIDE="cat head" \
    SUPPORT_LOCAL_TARGET_PROMPT_OVERRIDE="${local_prompt}" \
    TASKS="${TASKS_ONE}" \
    METHODS="${METHODS_ONE}" \
    SEEDS="${SEEDS_ONE}" \
    bash scripts/run_pretty_matrix.sh
}

prepare_profile "_profileD_mild_salience" \
  "${STRICT_PROMPT}" "${STRICT_LOCAL}" \
  "sunglass,lens,black,visible,eye" "sunglass,lens" \
  SUPPORT_V3_PROFILE_FACE_MIN_AREA_RATIO=0.008 \
  SUPPORT_V3_PROFILE_FACE_MAX_AREA_RATIO=0.045 \
  SUPPORT_V3_PROFILE_FACE_DILATE_RADIUS=2 \
  SUPPORT_V3_PROFILE_FACE_BLUR_KERNEL=3 \
  SUPPORT_V3_PROFILE_FACE_HEDIT=0.72 \
  SUPPORT_V3_PROFILE_FACE_TEXT=0.12 \
  SUPPORT_V3_PROFILE_FACE_RMS_TARGET=0.48 \
  SUPPORT_V3_PROFILE_FACE_REC_GUIDANCE_SCALE=0.26 \
  SUPPORT_V3_PROFILE_FACE_TRAJECTORY_PRESERVE_SCALE=0.16 \
  SUPPORT_V3_PROFILE_FACE_PRESERVE_DRIFT_BUDGET=0.12 \
  SUPPORT_V3_PROFILE_FACE_OUTSIDE_LOCK=0.92

prepare_profile "_profileE_balanced_salience" \
  "${STRICT_PROMPT}" "${STRICT_LOCAL}" \
  "sunglass,lens,black,visible,eye" "sunglass,lens" \
  SUPPORT_V3_PROFILE_FACE_MIN_AREA_RATIO=0.010 \
  SUPPORT_V3_PROFILE_FACE_MAX_AREA_RATIO=0.060 \
  SUPPORT_V3_PROFILE_FACE_DILATE_RADIUS=2 \
  SUPPORT_V3_PROFILE_FACE_BLUR_KERNEL=3 \
  SUPPORT_V3_PROFILE_FACE_HEDIT=0.78 \
  SUPPORT_V3_PROFILE_FACE_TEXT=0.16 \
  SUPPORT_V3_PROFILE_FACE_ANCHOR=0.02 \
  SUPPORT_V3_PROFILE_FACE_REGION=0.03 \
  SUPPORT_V3_PROFILE_FACE_RMS_TARGET=0.55 \
  SUPPORT_V3_PROFILE_FACE_EDIT_GAIN=2.4 \
  SUPPORT_V3_PROFILE_FACE_EDIT_WEIGHT_MAX=1.85 \
  SUPPORT_V3_PROFILE_FACE_REC_GUIDANCE_SCALE=0.24 \
  SUPPORT_V3_PROFILE_FACE_TRAJECTORY_PRESERVE_SCALE=0.14 \
  SUPPORT_V3_PROFILE_FACE_PRESERVE_DRIFT_BUDGET=0.14 \
  SUPPORT_V3_PROFILE_FACE_OUTSIDE_LOCK=0.90

prepare_profile "_profileF_strong_salience" \
  "${STRICT_PROMPT}" "${STRICT_LOCAL}" \
  "sunglass,lens,black,visible,eye" "sunglass,lens" \
  SUPPORT_V3_PROFILE_FACE_MIN_AREA_RATIO=0.012 \
  SUPPORT_V3_PROFILE_FACE_MAX_AREA_RATIO=0.075 \
  SUPPORT_V3_PROFILE_FACE_DILATE_RADIUS=3 \
  SUPPORT_V3_PROFILE_FACE_BLUR_KERNEL=3 \
  SUPPORT_V3_PROFILE_FACE_HEDIT=0.85 \
  SUPPORT_V3_PROFILE_FACE_TEXT=0.20 \
  SUPPORT_V3_PROFILE_FACE_ANCHOR=0.03 \
  SUPPORT_V3_PROFILE_FACE_REGION=0.05 \
  SUPPORT_V3_PROFILE_FACE_RMS_TARGET=0.62 \
  SUPPORT_V3_PROFILE_FACE_EDIT_GAIN=2.8 \
  SUPPORT_V3_PROFILE_FACE_EDIT_WEIGHT_MAX=2.10 \
  SUPPORT_V3_PROFILE_FACE_REC_GUIDANCE_SCALE=0.22 \
  SUPPORT_V3_PROFILE_FACE_TRAJECTORY_PRESERVE_SCALE=0.12 \
  SUPPORT_V3_PROFILE_FACE_PRESERVE_DRIFT_BUDGET=0.16 \
  SUPPORT_V3_PROFILE_FACE_OUTSIDE_LOCK=0.88

prepare_profile "_profileG_oval_lens_balanced" \
  "${OVAL_PROMPT}" "${OVAL_LOCAL}" \
  "black,oval,lens,eyepiece,visible,eye" "black,oval,lens,eyepiece" \
  SUPPORT_V3_PROFILE_FACE_MIN_AREA_RATIO=0.010 \
  SUPPORT_V3_PROFILE_FACE_MAX_AREA_RATIO=0.060 \
  SUPPORT_V3_PROFILE_FACE_DILATE_RADIUS=2 \
  SUPPORT_V3_PROFILE_FACE_BLUR_KERNEL=3 \
  SUPPORT_V3_PROFILE_FACE_HEDIT=0.78 \
  SUPPORT_V3_PROFILE_FACE_TEXT=0.16 \
  SUPPORT_V3_PROFILE_FACE_ANCHOR=0.02 \
  SUPPORT_V3_PROFILE_FACE_REGION=0.03 \
  SUPPORT_V3_PROFILE_FACE_RMS_TARGET=0.55 \
  SUPPORT_V3_PROFILE_FACE_EDIT_GAIN=2.4 \
  SUPPORT_V3_PROFILE_FACE_EDIT_WEIGHT_MAX=1.85 \
  SUPPORT_V3_PROFILE_FACE_REC_GUIDANCE_SCALE=0.24 \
  SUPPORT_V3_PROFILE_FACE_TRAJECTORY_PRESERVE_SCALE=0.14 \
  SUPPORT_V3_PROFILE_FACE_PRESERVE_DRIFT_BUDGET=0.14 \
  SUPPORT_V3_PROFILE_FACE_OUTSIDE_LOCK=0.90

echo "[profile-tune] manifest_count=$(wc -l < "${MANIFEST}")"

"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
methods = [
    "support_v3_controller_rmsgap_profileD_mild_salience",
    "support_v3_controller_rmsgap_profileE_balanced_salience",
    "support_v3_controller_rmsgap_profileF_strong_salience",
    "support_v3_controller_rmsgap_profileG_oval_lens_balanced",
]
missing = []
for method in methods:
    d = root / "outputs" / "pretty_matrix" / "cat_side_sunglasses_phase2" / method / "seed_10"
    absent = [name for name in ("result.png", "metadata.json", "stats.json", "command.txt") if not (d / name).is_file()]
    if absent:
        missing.append({"method": method, "missing": absent})
summary = {
    "complete": len(methods) - len(missing),
    "expected": len(methods),
    "missing_count": len(missing),
    "missing": missing,
}
run_dirs = sorted((root / "experiments" / "support_v3_2026-06-02").glob("phase2_profile_face_tune_seed10_20260605_*"))
run_dir = run_dirs[-1]
summary_path = run_dir / "phase2_profile_face_tune_seed10_completion.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[profile-tune] complete={summary['complete']}")
print(f"[profile-tune] expected={summary['expected']}")
print(f"[profile-tune] missing={summary['missing_count']}")
print(f"[profile-tune] summary={summary_path}")
if missing:
    raise SystemExit(1)
PY

tar -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  "experiments/support_v3_2026-06-02/${RUN_ID}" \
  "outputs/pretty_matrix/cat_side_sunglasses_phase2/support_v3_controller_rmsgap_profileD_mild_salience" \
  "outputs/pretty_matrix/cat_side_sunglasses_phase2/support_v3_controller_rmsgap_profileE_balanced_salience" \
  "outputs/pretty_matrix/cat_side_sunglasses_phase2/support_v3_controller_rmsgap_profileF_strong_salience" \
  "outputs/pretty_matrix/cat_side_sunglasses_phase2/support_v3_controller_rmsgap_profileG_oval_lens_balanced" \
  "data/paper_images/cat_on_grass.jpg"

echo "[profile-tune] artifact=${ARTIFACT}"
echo "[profile-tune] end=$(date -Is)"
