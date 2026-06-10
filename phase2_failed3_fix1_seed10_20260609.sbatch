#!/usr/bin/env bash
#SBATCH -J p2_f3_fix1_s10
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 03:00:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_failed3_fix1_seed10_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_TAG="phase2_failed3_fix1_seed10_20260609"
RUN_ID="${RUN_TAG}_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
COMPLETION="${RUN_DIR}/completion.json"
export RUN_ID

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[failed3-fix1] start=$(date -Is)"
echo "[failed3-fix1] host=$(hostname)"
echo "[failed3-fix1] job_id=${SLURM_JOB_ID:-none}"
echo "[failed3-fix1] run_dir=${RUN_DIR}"
nvidia-smi || true

export PYTHON="${ROOT}/.venv/bin/python"
export HF_HOME="${ROOT}/../.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MODEL_OFFLOAD=0
export LOW_VRAM=0
export NUM_INFERENCE_STEPS=28
export N_MAX=24
export MAX_IMAGE_SIZE=512
export SKIP_EXISTING=0
export REGENERATE_MASKS=1
export REUSE_SEMANTIC_MASKS=0
export METHOD_NAME_SUFFIX="${METHOD_NAME_SUFFIX:-_failed3_fix1_20260609}"

TASKS_FAILED3="
white_bowl_orange_tabletop_phase2
tshirt_star
yellow_vase_blue_phase2
"

METHODS_FAILED3="support_v3_controller_rmsgap"
SEEDS_FAILED3="10"

echo "[failed3-fix1] tasks=$(echo ${TASKS_FAILED3})"
echo "[failed3-fix1] methods=${METHODS_FAILED3}"
echo "[failed3-fix1] suffix=${METHOD_NAME_SUFFIX}"
echo "[failed3-fix1] seeds=${SEEDS_FAILED3}"

BATCH_MANIFEST="${MANIFEST}" \
PREPARE_ONLY=1 \
TASKS="${TASKS_FAILED3}" \
METHODS="${METHODS_FAILED3}" \
SEEDS="${SEEDS_FAILED3}" \
bash scripts/run_pretty_matrix.sh

echo "[failed3-fix1] manifest_count=$(wc -l < "${MANIFEST}")"
"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}"

"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
run_dir = root / "experiments" / "support_v3_2026-06-02" / os.environ["RUN_ID"]
suffix = os.environ.get("METHOD_NAME_SUFFIX", "")
tasks = ["white_bowl_orange_tabletop_phase2", "tshirt_star", "yellow_vase_blue_phase2"]
method = f"support_v3_controller_rmsgap{suffix}"
missing = []
for task in tasks:
    out = root / "outputs" / "pretty_matrix" / task / method / "seed_10"
    for name in ("result.png", "stats.json", "metadata.json", "command.txt"):
        if not (out / name).is_file():
            missing.append(str(out / name))
complete = len(tasks) - len({Path(p).parts[-4] for p in missing})
payload = {
    "scope": "Phase2 failed3 fix1 seed10",
    "method_suffix": suffix,
    "complete": complete,
    "expected": len(tasks),
    "missing_count": len(missing),
    "missing": missing,
}
(run_dir / "completion.json").write_text(json.dumps(payload, indent=2) + "\n")
print(f"[failed3-fix1] complete={payload['complete']} expected={payload['expected']} missing={payload['missing_count']}")
if missing:
    raise SystemExit(1)
PY

echo "[failed3-fix1] completion=${COMPLETION}"
echo "[failed3-fix1] end=$(date -Is)"
