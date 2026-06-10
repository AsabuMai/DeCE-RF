#!/usr/bin/env bash
#SBATCH -J t1t4_dcrf_3s
#SBATCH -p a100
#SBATCH -w a100-01
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH --mem=56G
#SBATCH -t 10:00:00
#SBATCH -o /cluster/users/grad/2025/25t8103/project/logs/phase2_t1_t4_decerf_seeds10_11_12_%j.log

set -euo pipefail

ROOT="/cluster/users/grad/2025/25t8103/project"
cd "${ROOT}"

RUN_TAG="phase2_t1_t4_decerf_seeds10_11_12_20260609"
RUN_ID="${RUN_TAG}_${SLURM_JOB_ID:-manual}"
RUN_DIR="${ROOT}/experiments/support_v3_2026-06-02/${RUN_ID}"
MANIFEST="${RUN_DIR}/command_manifest.txt"
SUMMARY="${RUN_DIR}/sd3_batch_summary.json"
COMPLETION="${RUN_DIR}/completion.json"
export RUN_ID

mkdir -p "${RUN_DIR}"
rm -f "${MANIFEST}"

echo "[t1-t4-decerf-3seed] start=$(date -Is)"
echo "[t1-t4-decerf-3seed] host=$(hostname)"
echo "[t1-t4-decerf-3seed] job_id=${SLURM_JOB_ID:-none}"
echo "[t1-t4-decerf-3seed] run_dir=${RUN_DIR}"
nvidia-smi || true

export PYTHON="${ROOT}/.venv/bin/python"
export HF_HOME="${ROOT}/../.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=0
export MODEL_OFFLOAD=0
export LOW_VRAM=0
export NUM_INFERENCE_STEPS=28
export N_MAX=24
export MAX_IMAGE_SIZE=512
export SKIP_EXISTING=1
export REGENERATE_MASKS=1
export REUSE_SEMANTIC_MASKS=0

# Keep this run separate from older formal/repair outputs unless explicitly overridden.
export METHOD_NAME_SUFFIX="${METHOD_NAME_SUFFIX:-_t1t4_3seed_20260609}"

TASKS_T1_T4="
cat_crown
dog_bow_tie_phase2
dog_front_sunglasses_phase2
bowl_apple_inside
white_bowl_orange_tabletop_phase2
brown_bowl_lemon_phase2
tshirt_star
mug_heart
tote_leaf
red_office_chair_to_blue_office_chair
green_mug_orange_phase2
yellow_vase_blue_phase2
"

METHODS_T1_T4="support_v3_controller_rmsgap"
SEEDS_T1_T4="10 11 12"

echo "[t1-t4-decerf-3seed] tasks=$(echo ${TASKS_T1_T4})"
echo "[t1-t4-decerf-3seed] methods=${METHODS_T1_T4}"
echo "[t1-t4-decerf-3seed] method_suffix=${METHOD_NAME_SUFFIX}"
echo "[t1-t4-decerf-3seed] seeds=${SEEDS_T1_T4}"
echo "[t1-t4-decerf-3seed] note=yellow_vase_blue_phase2 remains visually questionable; keep it flagged for human review."

BATCH_MANIFEST="${MANIFEST}" \
PREPARE_ONLY=1 \
TASKS="${TASKS_T1_T4}" \
METHODS="${METHODS_T1_T4}" \
SEEDS="${SEEDS_T1_T4}" \
bash scripts/run_pretty_matrix.sh

if [[ -s "${MANIFEST}" ]]; then
  echo "[t1-t4-decerf-3seed] manifest_count=$(wc -l < "${MANIFEST}")"
  "${PYTHON}" scripts/run_sd3_batch.py \
    --manifest "${MANIFEST}" \
    --summary-output "${SUMMARY}" \
    --skip-existing
else
  echo "[t1-t4-decerf-3seed] manifest_count=0 all complete"
  printf '{"records":[],"note":"all requested runs already complete"}\n' > "${SUMMARY}"
fi

"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path("/cluster/users/grad/2025/25t8103/project")
run_id = os.environ.get("RUN_ID")
run_dir = root / "experiments" / "support_v3_2026-06-02" / run_id
suffix = os.environ.get("METHOD_NAME_SUFFIX", "")

tasks = """
cat_crown
dog_bow_tie_phase2
dog_front_sunglasses_phase2
bowl_apple_inside
white_bowl_orange_tabletop_phase2
brown_bowl_lemon_phase2
tshirt_star
mug_heart
tote_leaf
red_office_chair_to_blue_office_chair
green_mug_orange_phase2
yellow_vase_blue_phase2
""".split()
methods = [f"support_v3_controller_rmsgap{suffix}"]
seeds = "10 11 12".split()

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

summary = {
    "scope": "Phase2 T1-T4 DeCE-RF only, seeds 10/11/12",
    "method_suffix": suffix,
    "complete": complete,
    "expected": len(tasks) * len(methods) * len(seeds),
    "missing_count": len(missing),
    "yellow_vase_blue_phase2_status": "flag_for_human_review",
    "missing": missing,
}
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "completion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(run_dir / "completion.txt").write_text(
    f"complete={summary['complete']}\nexpected={summary['expected']}\nmissing={summary['missing_count']}\n",
    encoding="utf-8",
)
print(f"[t1-t4-decerf-3seed] complete={summary['complete']}")
print(f"[t1-t4-decerf-3seed] expected={summary['expected']}")
print(f"[t1-t4-decerf-3seed] missing={summary['missing_count']}")
print(f"[t1-t4-decerf-3seed] completion={run_dir / 'completion.json'}")
if missing:
    raise SystemExit(1)
PY

echo "[t1-t4-decerf-3seed] end=$(date -Is)"
