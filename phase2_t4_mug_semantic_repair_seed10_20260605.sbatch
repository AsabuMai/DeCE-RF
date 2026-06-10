#!/usr/bin/env bash
#SBATCH -J t4_mug_fix
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 01:00:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t4_mug_semantic_repair_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_t4_mug_semantic_repair_seed10_20260605_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
ARTIFACT="${RUN_DIR}/phase2_t4_mug_semantic_repair_seed10_${SLURM_JOB_ID:-manual}.tgz"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[t4-mug-repair] start=$(date -Is)"
echo "[t4-mug-repair] host=$(hostname)"
echo "[t4-mug-repair] job_id=${SLURM_JOB_ID:-none}"
echo "[t4-mug-repair] run_dir=${RUN_DIR}"
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

TASKS_ONE="green_mug_orange_phase2"
METHODS_ONE="support_v3_controller_rmsgap"
SEEDS_ONE="10"

BACKUP_DIR="${ROOT}/outputs/pretty_matrix_registry_backup/t4_mug_semantic_repair_seed10_${SLURM_JOB_ID:-manual}"
src="${ROOT}/outputs/pretty_matrix/green_mug_orange_phase2/support_v3_controller_rmsgap/seed_10"
if [[ -d "${src}" ]]; then
  mkdir -p "${BACKUP_DIR}/green_mug_orange_phase2/support_v3_controller_rmsgap"
  rm -rf "${BACKUP_DIR}/green_mug_orange_phase2/support_v3_controller_rmsgap/seed_10"
  mv "${src}" "${BACKUP_DIR}/green_mug_orange_phase2/support_v3_controller_rmsgap/seed_10"
  echo "[t4-mug-repair] backup=${BACKUP_DIR}"
fi

BATCH_MANIFEST="${MANIFEST}" \
PREPARE_ONLY=1 \
SKIP_EXISTING=0 \
TASKS="${TASKS_ONE}" \
METHODS="${METHODS_ONE}" \
SEEDS="${SEEDS_ONE}" \
bash scripts/run_pretty_matrix.sh

echo "[t4-mug-repair] manifest_count=$(wc -l < "${MANIFEST}")"

"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
d = root / "outputs" / "pretty_matrix" / "green_mug_orange_phase2" / "support_v3_controller_rmsgap" / "seed_10"
missing = [name for name in ("result.png", "metadata.json", "stats.json", "command.txt") if not (d / name).is_file()]
summary = {"task": "green_mug_orange_phase2", "complete": not missing, "missing": missing}
run_dirs = sorted((root / "experiments" / "support_v3_2026-06-02").glob("phase2_t4_mug_semantic_repair_seed10_20260605_*"))
summary_path = run_dirs[-1] / "phase2_t4_mug_semantic_repair_seed10_completion.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[t4-mug-repair] complete={summary['complete']}")
print(f"[t4-mug-repair] summary={summary_path}")
if missing:
    raise SystemExit(1)
PY

tar -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  "experiments/support_v3_2026-06-02/${RUN_ID}" \
  "outputs/pretty_matrix/green_mug_orange_phase2/support_v3_controller_rmsgap"

echo "[t4-mug-repair] artifact=${ARTIFACT}"
echo "[t4-mug-repair] end=$(date -Is)"
