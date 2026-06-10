#!/usr/bin/env bash
#SBATCH -J phase2_t1_g10
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 08:00:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_e1_seed10_t1_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_ID="phase2_e1_seed10_t1_20260605_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
COMPLETION="${RUN_DIR}/completion.txt"

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[phase2-e1-gate] start=$(date -Is)"
echo "[phase2-e1-gate] host=$(hostname)"
echo "[phase2-e1-gate] cwd=$(pwd)"
echo "[phase2-e1-gate] job_id=${SLURM_JOB_ID:-none}"
echo "[phase2-e1-gate] run_dir=${RUN_DIR}"
echo "[phase2-e1-gate] manifest=${MANIFEST}"
echo "[phase2-e1-gate] cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

export PYTHON="${ROOT}/.venv/bin/python"
export HF_HOME="${ROOT}/../.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MODEL_OFFLOAD=0
export LOW_VRAM=0
export NUM_INFERENCE_STEPS=28
export N_MAX=24
export MAX_IMAGE_SIZE=512

TASKS_PHASE2_E1="
cat_crown
dog_bow_tie_phase2
dog_front_sunglasses_phase2
"

METHODS_PHASE2_E1="
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
"

SEEDS_PHASE2_E1="10"

echo "[phase2-e1-gate] tasks=$(echo ${TASKS_PHASE2_E1})"
echo "[phase2-e1-gate] methods=$(echo ${METHODS_PHASE2_E1})"
echo "[phase2-e1-gate] seeds=${SEEDS_PHASE2_E1}"

echo "[phase2-e1-gate] preparing commands and helper artifacts"
BATCH_MANIFEST="${MANIFEST}" \
PREPARE_ONLY=1 \
SKIP_EXISTING=0 \
TASKS="${TASKS_PHASE2_E1}" \
METHODS="${METHODS_PHASE2_E1}" \
SEEDS="${SEEDS_PHASE2_E1}" \
bash scripts/run_pretty_matrix.sh

echo "[phase2-e1-gate] manifest_count=$(wc -l < "${MANIFEST}")"

echo "[phase2-e1-gate] running SD3 batch with one loaded pipeline"
"${PYTHON}" scripts/run_sd3_batch.py \
  --manifest "${MANIFEST}" \
  --summary-output "${SUMMARY}" \
  --skip-existing

echo "[phase2-e1-gate] checking completion"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
run_dirs = sorted((root / "experiments" / "support_v3_2026-06-02").glob("phase2_e1_seed10_t1_20260605_*"))
run_dir = run_dirs[-1]

tasks = """
cat_crown
dog_bow_tie_phase2
dog_front_sunglasses_phase2
""".split()
methods = """
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
""".split()
seeds = ["10"]

missing = []
complete = 0
for task in tasks:
    for method in methods:
        for seed in seeds:
            d = root / "outputs" / "pretty_matrix" / task / method / f"seed_{seed}"
            absent = [name for name in ("result.png", "metadata.json", "stats.json", "command.txt") if not (d / name).is_file()]
            if absent:
                missing.append({"task": task, "method": method, "seed": seed, "missing": absent})
            else:
                complete += 1

summary_path = run_dir / "phase2_e1_seed10_t1_completion.json"
summary = {
    "complete": complete,
    "expected": len(tasks) * len(methods) * len(seeds),
    "missing_count": len(missing),
    "missing": missing,
}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
(run_dir / "completion.txt").write_text(
    f"complete={complete}\nexpected={summary['expected']}\nmissing={len(missing)}\n",
    encoding="utf-8",
)
print(f"[phase2-e1-gate] complete={complete}")
print(f"[phase2-e1-gate] expected={summary['expected']}")
print(f"[phase2-e1-gate] missing={len(missing)}")
print(f"[phase2-e1-gate] summary={summary_path}")
PY

echo "[phase2-e1-gate] end=$(date -Is)"
