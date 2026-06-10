#!/usr/bin/env bash
#SBATCH -J phase2_t3_reg
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 02:00:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t3_registry_decerf_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_t3_registry_decerf_seed10_20260605_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
ARTIFACT="${ROOT}/experiments/support_v3_2026-06-02/phase2_t3_registry_decerf_seed10_${SLURM_JOB_ID:-manual}.tgz"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[phase2-t3-registry] start=$(date -Is)"
echo "[phase2-t3-registry] host=$(hostname)"
echo "[phase2-t3-registry] cwd=$(pwd)"
echo "[phase2-t3-registry] job_id=${SLURM_JOB_ID:-none}"
echo "[phase2-t3-registry] run_dir=${RUN_DIR}"
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

TASKS_PHASE2_T3="
tshirt_star
mug_heart
tote_leaf
"
METHODS_PHASE2_T3="support_v3_controller_rmsgap"
SEEDS_PHASE2_T3="10"

BACKUP_DIR="${ROOT}/outputs/pretty_matrix_registry_backup/t3_decerf_seed10_${SLURM_JOB_ID:-manual}"
mkdir -p "${BACKUP_DIR}"
for task in ${TASKS_PHASE2_T3}; do
  src="${ROOT}/outputs/pretty_matrix/${task}/support_v3_controller_rmsgap/seed_10"
  if [[ -d "${src}" ]]; then
    mkdir -p "${BACKUP_DIR}/${task}/support_v3_controller_rmsgap"
    rm -rf "${BACKUP_DIR}/${task}/support_v3_controller_rmsgap/seed_10"
    mv "${src}" "${BACKUP_DIR}/${task}/support_v3_controller_rmsgap/seed_10"
    echo "[phase2-t3-registry] backed_up=${src} -> ${BACKUP_DIR}/${task}/support_v3_controller_rmsgap/seed_10"
  fi
done

echo "[phase2-t3-registry] tasks=$(echo ${TASKS_PHASE2_T3})"
echo "[phase2-t3-registry] methods=${METHODS_PHASE2_T3}"
echo "[phase2-t3-registry] seeds=${SEEDS_PHASE2_T3}"

BATCH_MANIFEST="${MANIFEST}" \
PREPARE_ONLY=1 \
SKIP_EXISTING=0 \
TASKS="${TASKS_PHASE2_T3}" \
METHODS="${METHODS_PHASE2_T3}" \
SEEDS="${SEEDS_PHASE2_T3}" \
bash scripts/run_pretty_matrix.sh

echo "[phase2-t3-registry] manifest_count=$(wc -l < "${MANIFEST}")"

"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
tasks = "tshirt_star mug_heart tote_leaf".split()
missing = []
for task in tasks:
    d = root / "outputs" / "pretty_matrix" / task / "support_v3_controller_rmsgap" / "seed_10"
    absent = [name for name in ("result.png", "metadata.json", "stats.json", "command.txt") if not (d / name).is_file()]
    if absent:
        missing.append({"task": task, "missing": absent})
summary = {"complete": len(tasks) - len(missing), "expected": len(tasks), "missing_count": len(missing), "missing": missing}
run_dirs = sorted((root / "experiments" / "support_v3_2026-06-02").glob("phase2_t3_registry_decerf_seed10_20260605_*"))
run_dir = run_dirs[-1]
summary_path = run_dir / "phase2_t3_registry_decerf_seed10_completion.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"[phase2-t3-registry] complete={summary['complete']}")
print(f"[phase2-t3-registry] expected={summary['expected']}")
print(f"[phase2-t3-registry] missing={summary['missing_count']}")
print(f"[phase2-t3-registry] summary={summary_path}")
if missing:
    raise SystemExit(1)
PY

tar -czf "${ARTIFACT}" \
  -C "${ROOT}" \
  "experiments/support_v3_2026-06-02/${RUN_ID}" \
  "outputs/pretty_matrix/tshirt_star/support_v3_controller_rmsgap" \
  "outputs/pretty_matrix/mug_heart/support_v3_controller_rmsgap" \
  "outputs/pretty_matrix/tote_leaf/support_v3_controller_rmsgap"

echo "[phase2-t3-registry] artifact=${ARTIFACT}"
echo "[phase2-t3-registry] end=$(date -Is)"
