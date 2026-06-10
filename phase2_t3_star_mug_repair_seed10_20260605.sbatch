#!/usr/bin/env bash
#SBATCH -J phase2_t3_sm
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 01:30:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t3_star_mug_repair_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_t3_star_mug_repair_seed10_20260605_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
ARTIFACT="${ROOT}/experiments/support_v3_2026-06-02/phase2_t3_star_mug_repair_seed10_${SLURM_JOB_ID:-manual}.tgz"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[t3-star-mug-repair] start=$(date -Is)"
echo "[t3-star-mug-repair] host=$(hostname)"
echo "[t3-star-mug-repair] job_id=${SLURM_JOB_ID:-none}"
echo "[t3-star-mug-repair] run_dir=${RUN_DIR}"
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

TASKS_REPAIR="tshirt_star mug_heart"
BACKUP="${ROOT}/outputs/pretty_matrix_registry_backup/t3_star_mug_repair_seed10_${SLURM_JOB_ID:-manual}"
for task in ${TASKS_REPAIR}; do
  src="${ROOT}/outputs/pretty_matrix/${task}/support_v3_controller_rmsgap/seed_10"
  if [[ -d "${src}" ]]; then
    mkdir -p "${BACKUP}/${task}/support_v3_controller_rmsgap"
    rm -rf "${BACKUP}/${task}/support_v3_controller_rmsgap/seed_10"
    mv "${src}" "${BACKUP}/${task}/support_v3_controller_rmsgap/seed_10"
    echo "[t3-star-mug-repair] backed_up=${src}"
  fi
done

BATCH_MANIFEST="${MANIFEST}" \
PREPARE_ONLY=1 \
SKIP_EXISTING=0 \
TASKS="${TASKS_REPAIR}" \
METHODS="support_v3_controller_rmsgap" \
SEEDS="10" \
bash scripts/run_pretty_matrix.sh

echo "[t3-star-mug-repair] manifest_count=$(wc -l < "${MANIFEST}")"

"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}"

"${PYTHON}" - <<'PY'
from pathlib import Path
root = Path("/cluster/users/grad/2025/25t8103/project")
missing = []
for task in "tshirt_star mug_heart".split():
    d = root / "outputs" / "pretty_matrix" / task / "support_v3_controller_rmsgap" / "seed_10"
    absent = [name for name in ("result.png", "metadata.json", "stats.json", "command.txt") if not (d / name).is_file()]
    if absent:
        missing.append({"task": task, "missing": absent})
print(f"[t3-star-mug-repair] missing={missing}")
if missing:
    raise SystemExit(1)
PY

tar -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  "experiments/support_v3_2026-06-02/${RUN_ID}" \
  "outputs/pretty_matrix/tshirt_star/support_v3_controller_rmsgap" \
  "outputs/pretty_matrix/mug_heart/support_v3_controller_rmsgap"

echo "[t3-star-mug-repair] artifact=${ARTIFACT}"
echo "[t3-star-mug-repair] end=$(date -Is)"
