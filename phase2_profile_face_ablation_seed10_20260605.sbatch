#!/usr/bin/env bash
#SBATCH -J phase2_prof_ab
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 02:00:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_profile_face_ablation_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_profile_face_ablation_seed10_20260605_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[profile-ablation] start=$(date -Is)"
echo "[profile-ablation] host=$(hostname)"
echo "[profile-ablation] cwd=$(pwd)"
echo "[profile-ablation] job_id=${SLURM_JOB_ID:-none}"
echo "[profile-ablation] run_dir=${RUN_DIR}"
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
OLD_PROMPT="A side-view photo of the same cat standing in the same green grass, wearing small black sunglasses fitted over its visible eye and nose bridge, while the fur, pose, grass, and lighting remain unchanged."
STRICT_PROMPT="A strict side-profile photo of the same cat facing right in the same green grass, with only the visible eye area edited to have a small black side-view sunglass lens over its visible eye, while the head direction, one-eye-visible profile, fur outline, pose, grass, and lighting remain unchanged."
OLD_LOCAL="A close-up side-view photo of small black sunglasses fitted over a cat's visible eye and nose bridge."
STRICT_LOCAL="A close-up strict side-profile photo of one small black side-view sunglass lens over the single visible eye of a cat."

echo "[profile-ablation] preparing A current_on_face_old_prompt"
METHOD_NAME_SUFFIX="_profileA_old_onface" \
BATCH_MANIFEST="${MANIFEST}" \
PREPARE_ONLY=1 \
SKIP_EXISTING=0 \
TARGET_PROMPT_OVERRIDE="${OLD_PROMPT}" \
ATTENTION_TARGET_WORDS_OVERRIDE="sunglasses,eye,face,cat" \
CHANGED_TARGET_WORDS_OVERRIDE="sunglasses" \
SUPPORT_NEW_TOKENS_OVERRIDE="sunglasses,glasses" \
SUPPORT_HOST_TOKENS_OVERRIDE="cat,face,eye" \
SUPPORT_V3_RELATION_OVERRIDE="on_face" \
SEMANTIC_PHRASE_OVERRIDE="cat" \
SUPPORT_LOCAL_TARGET_PROMPT_OVERRIDE="${OLD_LOCAL}" \
TASKS="${TASKS_ONE}" \
METHODS="${METHODS_ONE}" \
SEEDS="${SEEDS_ONE}" \
bash scripts/run_pretty_matrix.sh

echo "[profile-ablation] preparing B on_face_strict_profile_prompt"
METHOD_NAME_SUFFIX="_profileB_strict_onface" \
BATCH_MANIFEST="${MANIFEST}" \
PREPARE_ONLY=1 \
SKIP_EXISTING=0 \
TARGET_PROMPT_OVERRIDE="${STRICT_PROMPT}" \
ATTENTION_TARGET_WORDS_OVERRIDE="sunglass,lens,visible,eye" \
CHANGED_TARGET_WORDS_OVERRIDE="sunglass,lens" \
SUPPORT_NEW_TOKENS_OVERRIDE="sunglass,lens" \
SUPPORT_HOST_TOKENS_OVERRIDE="cat,face,eye" \
SUPPORT_V3_RELATION_OVERRIDE="on_face" \
SEMANTIC_PHRASE_OVERRIDE="cat" \
SUPPORT_LOCAL_TARGET_PROMPT_OVERRIDE="${STRICT_LOCAL}" \
TASKS="${TASKS_ONE}" \
METHODS="${METHODS_ONE}" \
SEEDS="${SEEDS_ONE}" \
bash scripts/run_pretty_matrix.sh

echo "[profile-ablation] preparing C on_profile_face_strict_prompt"
METHOD_NAME_SUFFIX="_profileC_onprofile_strict" \
BATCH_MANIFEST="${MANIFEST}" \
PREPARE_ONLY=1 \
SKIP_EXISTING=0 \
TARGET_PROMPT_OVERRIDE="${STRICT_PROMPT}" \
TASKS="${TASKS_ONE}" \
METHODS="${METHODS_ONE}" \
SEEDS="${SEEDS_ONE}" \
bash scripts/run_pretty_matrix.sh

echo "[profile-ablation] manifest_count=$(wc -l < "${MANIFEST}")"

"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
methods = [
    "support_v3_controller_rmsgap_profileA_old_onface",
    "support_v3_controller_rmsgap_profileB_strict_onface",
    "support_v3_controller_rmsgap_profileC_onprofile_strict",
]
missing = []
for method in methods:
    d = root / "outputs" / "pretty_matrix" / "cat_side_sunglasses_phase2" / method / "seed_10"
    absent = [name for name in ("result.png", "metadata.json", "stats.json", "command.txt") if not (d / name).is_file()]
    if absent:
        missing.append({"method": method, "missing": absent})
summary = {"complete": len(methods) - len(missing), "expected": len(methods), "missing_count": len(missing), "missing": missing}
run_dirs = sorted((root / "experiments" / "support_v3_2026-06-02").glob("phase2_profile_face_ablation_seed10_20260605_*"))
run_dir = run_dirs[-1]
summary_path = run_dir / "phase2_profile_face_ablation_seed10_completion.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[profile-ablation] complete={summary['complete']}")
print(f"[profile-ablation] expected={summary['expected']}")
print(f"[profile-ablation] missing={summary['missing_count']}")
print(f"[profile-ablation] summary={summary_path}")
if missing:
    raise SystemExit(1)
PY

echo "[profile-ablation] end=$(date -Is)"
